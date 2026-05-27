using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.Domain;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Moq;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace AnomalyDetection.Tests.Integration
{
    public class InferenceTestFactory : WebApplicationFactory<Program>
    {
        private readonly string _dbName = "InferenceTestDb_" + Guid.NewGuid();

        public Mock<IRouterService> RouterMock { get; } = new Mock<IRouterService>();
        public Mock<IModelManagerService> ModelManagerMock { get; } = new Mock<IModelManagerService>();

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            builder.UseEnvironment("Testing");

            builder.ConfigureServices(services =>
            {
                var modelManagerDescriptor = services.SingleOrDefault(
                    d => d.ServiceType == typeof(IModelManagerService));
                if (modelManagerDescriptor != null)
                    services.Remove(modelManagerDescriptor);

                var routerDescriptor = services.SingleOrDefault(
                    d => d.ServiceType == typeof(IRouterService));
                if (routerDescriptor != null)
                    services.Remove(routerDescriptor);

                services.AddSingleton<IModelManagerService>(_ => ModelManagerMock.Object);
                services.AddSingleton<IRouterService>(_ => RouterMock.Object);

                services.AddDbContext<AppDbContext>(options =>
                    options.UseInMemoryDatabase(_dbName));
            });
        }
    }

    public class InferenceIntegrationTests : IClassFixture<InferenceTestFactory>, IDisposable
    {
        #region Setup
        private readonly InferenceTestFactory _factory;
        private readonly HttpClient _client;

        public InferenceIntegrationTests(InferenceTestFactory factory)
        {
            _factory = factory;
            _client = factory.CreateClient();
        }

        public void Dispose()
        {
            _client.Dispose();
        }

        private static StringContent JsonContent(object obj) =>
            new StringContent(JsonSerializer.Serialize(obj), Encoding.UTF8, "application/json");

        private async Task<string> GetTokenAsync(string role = "User")
        {
            var username = "user_" + Guid.NewGuid().ToString("N")[..8];
            var password = "Password123!";
            var email = username + "@test.com";

            await _client.PostAsync("/api/v1/auth/register", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            if (role == "Admin")
            {
                using var scope = _factory.Services.CreateScope();
                var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                var u = db.Users.First(x => x.Username == username);
                u.Role = "Admin";
                db.SaveChanges();
            }

            var loginResponse = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            var body = await loginResponse.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);
            return json.GetProperty("token").GetString()!;
        }

        private HttpClient CreateAuthenticatedClient(string token)
        {
            var client = _factory.CreateClient();
            client.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
            return client;
        }

        private static MultipartFormDataContent CreateImageFormContent(
            string fileName = "test.png",
            byte[]? fileContent = null)
        {
            var content = new MultipartFormDataContent();
            var imageContent = new ByteArrayContent(fileContent ?? new byte[] { 1, 2, 3, 4, 5 });
            imageContent.Headers.ContentType = new MediaTypeHeaderValue("image/png");
            imageContent.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
            {
                Name = "image",
                FileName = fileName
            };
            content.Add(imageContent);
            return content;
        }
        private void SetupSuccessfulPipeline(
            string category = "bottle",
            bool isAnomaly = false,
            float score = 0.3f,
            float threshold = 0.5f)
        {
            var mockAnomalyService = new Mock<IAnomalyDetectionService>();
            var metadata = new ModelMetadata
            {
                ClassName = category,
                OptimalThreshold = threshold,
                ScoreMin = 0f,
                ScoreMax = 1f,
                ApplyMask = true,
                HeatmapUseGlobalMax = false
            };
            var anomalyResult = new AnomalyResult
            {
                IsAnomaly = isAnomaly,
                Score = score,
                UsedThreshold = threshold,
                HeatmapBase64 = null
            };

            _factory.RouterMock
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync((category, 0.95f));

            _factory.ModelManagerMock
                .Setup(m => m.GetModelForCategory(category))
                .Returns((mockAnomalyService.Object, metadata));

            mockAnomalyService
                .Setup(s => s.PredictAnomalyScore(
                    It.IsAny<Stream>(), It.IsAny<float>(), It.IsAny<float>(), It.IsAny<float>(),
                    It.IsAny<bool>(), It.IsAny<bool>(), It.IsAny<bool>()))
                .Returns(anomalyResult);
        }
        #endregion

        #region Authorization Tests
        [Fact]
        public async Task DetectAnomaly_Returns401_WhenNoTokenProvided()
        {
            // Arrange
            var content = CreateImageFormContent();

            // Act
            var response = await _client.PostAsync("/api/v1/inference/detect_anomaly", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task DetectAnomaly_Returns200_WhenUserIsAuthenticated()
        {
            // Arrange
            SetupSuccessfulPipeline();
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task DetectAnomaly_Returns200_WhenAdminIsAuthenticated()
        {
            // Arrange
            SetupSuccessfulPipeline();
            var token = await GetTokenAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }
        #endregion

        #region Validation Tests
        [Fact]
        public async Task DetectAnomaly_Returns400_WhenNoImageUploaded()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);
            var emptyContent = new MultipartFormDataContent();

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly", emptyContent);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }
        #endregion

        #region Router Rejection Tests
        [Fact]
        public async Task DetectAnomaly_Returns400_WhenRouterRejectsImage()
        {
            // Arrange
            _factory.RouterMock
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("unknown", 0.05f));

            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("Image not recognized");
        }

        [Fact]
        public async Task DetectAnomaly_Returns400_WithConfidenceInfo_WhenRouterRejects()
        {
            // Arrange
            _factory.RouterMock
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("unknown", 0.12f));

            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("12");
        }
        #endregion

        #region Model Not Found Tests
        [Fact]
        public async Task DetectAnomaly_Returns404_WhenModelBankIsMissing()
        {
            // Arrange
            _factory.RouterMock
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("bottle", 0.95f));

            _factory.ModelManagerMock
                .Setup(m => m.GetModelForCategory("bottle"))
                .Throws(new FileNotFoundException("Memory bank not found"));

            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.NotFound);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("AI model for this category is currently unavailable");
        }
        #endregion

        #region Successful Inference Tests
        [Fact]
        public async Task DetectAnomaly_ReturnsCorrectCategory_InResponse()
        {
            // Arrange
            SetupSuccessfulPipeline(category: "capsule");
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("predictedCategory").GetString().Should().Be("capsule");
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsIsAnomaly_WhenDefectDetected()
        {
            // Arrange
            SetupSuccessfulPipeline(isAnomaly: true, score: 0.9f);
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("isAnomaly").GetBoolean().Should().BeTrue();
            json.GetProperty("score").GetSingle().Should().Be(0.9f);
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsIsNotAnomaly_WhenNormalImage()
        {
            // Arrange
            SetupSuccessfulPipeline(isAnomaly: false, score: 0.1f);
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("isAnomaly").GetBoolean().Should().BeFalse();
            json.GetProperty("score").GetSingle().Should().Be(0.1f);
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsUsedThreshold_InResponse()
        {
            // Arrange
            SetupSuccessfulPipeline(threshold: 0.5f);
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("usedThreshold").GetSingle().Should().Be(0.5f);
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsNullHeatmap_WhenReturnHeatmapIsFalse()
        {
            // Arrange
            SetupSuccessfulPipeline();
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            if (json.TryGetProperty("heatmapBase64", out var heatmap))
                heatmap.ValueKind.Should().Be(JsonValueKind.Null);
        }

        [Fact]
        public async Task DetectAnomaly_SavesInferenceRecord_ToDatabase()
        {
            // Arrange
            SetupSuccessfulPipeline(category: "screw", isAnomaly: true, score: 0.8f);
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            var countBefore = 0;
            using (var scope0 = _factory.Services.CreateScope())
            {
                var db0 = scope0.ServiceProvider.GetRequiredService<AppDbContext>();
                countBefore = db0.InferenceRecords.Count(r => r.Category == "screw");
            }

            // Act
            await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent("part.png"));

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var countAfter = db.InferenceRecords.Count(r => r.Category == "screw");

            countAfter.Should().Be(countBefore + 1);

            var record = db.InferenceRecords.LastOrDefault(r => r.Category == "screw");
            record.Should().NotBeNull();
            record!.IsAnomaly.Should().BeTrue();
            record.ImageName.Should().Be("part.png");
        }

        [Fact]
        public async Task DetectAnomaly_DoesNotSaveRecord_WhenRouterRejectsImage()
        {
            // Arrange
            _factory.RouterMock
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("unknown", 0.05f));

            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            var countBefore = 0;
            using (var scope = _factory.Services.CreateScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                countBefore = db.InferenceRecords.Count();
            }

            // Act
            await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            using var scope2 = _factory.Services.CreateScope();
            var db2 = scope2.ServiceProvider.GetRequiredService<AppDbContext>();
            db2.InferenceRecords.Count().Should().Be(countBefore);
        }
        #endregion

        #region Internal Server Error Tests
        [Fact]
        public async Task DetectAnomaly_Returns500_WhenUnexpectedErrorOccurs()
        {
            // Arrange
            _factory.RouterMock
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ThrowsAsync(new Exception("Unexpected internal error"));

            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PostAsync("/api/v1/inference/detect_anomaly",
                CreateImageFormContent());

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
        }
        #endregion
    }
}