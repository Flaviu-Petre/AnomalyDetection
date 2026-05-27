using AnomalyDetection.Api.Data;
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
    public class FeedbackTestFactory : WebApplicationFactory<Program>
    {
        private readonly string _dbName = "FeedbackTestDb_" + Guid.NewGuid();

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

                services.AddSingleton<IModelManagerService>(_ => Mock.Of<IModelManagerService>());
                services.AddSingleton<IRouterService>(_ => Mock.Of<IRouterService>());

                services.AddDbContext<AppDbContext>(options =>
                    options.UseInMemoryDatabase(_dbName));
            });
        }
    }

    public class FeedbackIntegrationTests : IClassFixture<FeedbackTestFactory>, IDisposable
    {
        #region Setup
        private readonly FeedbackTestFactory _factory;
        private readonly HttpClient _client;
        private readonly string _testFeedbackDir;

        public FeedbackIntegrationTests(FeedbackTestFactory factory)
        {
            _factory = factory;
            _client = factory.CreateClient();
            _testFeedbackDir = Path.Combine(Directory.GetCurrentDirectory(), "FeedbackData");
        }

        public void Dispose()
        {
            _client.Dispose();

            if (Directory.Exists(_testFeedbackDir))
                Directory.Delete(_testFeedbackDir, recursive: true);
        }

        private static StringContent JsonContent(object obj) =>
            new StringContent(JsonSerializer.Serialize(obj), Encoding.UTF8, "application/json");

        private async Task<(string Token, int UserId)> RegisterAndLoginAsync(string role = "User")
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
            var token = json.GetProperty("token").GetString()!;
            var userId = GetUserIdFromToken(token);

            return (token, userId);
        }

        private HttpClient CreateAuthenticatedClient(string token)
        {
            var client = _factory.CreateClient();
            client.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
            return client;
        }

        private static int GetUserIdFromToken(string token)
        {
            var parts = token.Split('.');
            var payload = parts[1];
            var padded = payload.PadRight(payload.Length + (4 - payload.Length % 4) % 4, '=');
            var json = JsonSerializer.Deserialize<JsonElement>(
                Encoding.UTF8.GetString(Convert.FromBase64String(padded)));
            return int.Parse(json.GetProperty("id").GetString()!);
        }

        private static MultipartFormDataContent CreateFeedbackFormContent(
            string category = "bottle",
            bool isActuallyAnomaly = true,
            string fileName = "test.png",
            byte[]? fileContent = null)
        {
            var content = new MultipartFormDataContent();
            content.Add(new StringContent(category), "category");
            content.Add(new StringContent(isActuallyAnomaly.ToString().ToLower()), "isActuallyAnomaly");

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
        #endregion

        #region SubmitFeedback Tests
        [Fact]
        public async Task SubmitFeedback_Returns401_WhenNoTokenProvided()
        {
            // Arrange
            var content = CreateFeedbackFormContent();

            // Act
            var response = await _client.PostAsync("/api/v1/feedback", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task SubmitFeedback_Returns200_WhenRequestIsValid()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);
            var content = CreateFeedbackFormContent("bottle", true);

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task SubmitFeedback_Returns200_ForGoodImage()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);
            var content = CreateFeedbackFormContent("capsule", false);

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task SubmitFeedback_Returns400_WhenCategoryIsMissing()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);
            var content = CreateFeedbackFormContent(category: "");

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task SubmitFeedback_Returns400_WhenImageIsMissing()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            var content = new MultipartFormDataContent();
            content.Add(new StringContent("bottle"), "category");
            content.Add(new StringContent("true"), "isActuallyAnomaly");

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("Image file is required");
        }

        [Fact]
        public async Task SubmitFeedback_SavesFile_ToCorrectDirectory()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);
            var content = CreateFeedbackFormContent("bottle", true);

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var savedPath = json.GetProperty("savedPath").GetString();
            savedPath.Should().Contain("bottle");
            savedPath.Should().Contain("anomaly");
            File.Exists(savedPath).Should().BeTrue();
        }

        [Fact]
        public async Task SubmitFeedback_SavesFile_ToGoodDirectory_WhenNotAnomaly()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);
            var content = CreateFeedbackFormContent("bottle", false);

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var savedPath = json.GetProperty("savedPath").GetString();
            savedPath.Should().Contain("good");
            File.Exists(savedPath).Should().BeTrue();
        }

        [Fact]
        public async Task SubmitFeedback_NormalizesCategory_ToLowercase()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);
            var content = CreateFeedbackFormContent("BOTTLE", true);

            // Act
            var response = await client.PostAsync("/api/v1/feedback", content);
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var savedPath = json.GetProperty("savedPath").GetString();
            savedPath.Should().Contain("bottle");
            savedPath.Should().NotContain("BOTTLE");
        }
        #endregion

        #region GetFeedbackSummary Tests
        [Fact]
        public async Task GetFeedbackSummary_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/feedback/summary");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task GetFeedbackSummary_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/summary");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task GetFeedbackSummary_Returns200_WhenUserIsAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/summary");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetFeedbackSummary_ReturnsSummary_AfterFeedbackSubmitted()
        {
            // Arrange
            var (userToken, _) = await RegisterAndLoginAsync("User");
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var userClient = CreateAuthenticatedClient(userToken);
            var adminClient = CreateAuthenticatedClient(adminToken);

            await userClient.PostAsync("/api/v1/feedback",
                CreateFeedbackFormContent("capsule", true));

            // Act
            var response = await adminClient.GetAsync("/api/v1/feedback/summary");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetArrayLength().Should().BeGreaterThanOrEqualTo(1);
        }
        #endregion

        #region GetFeedbackImageList Tests
        [Fact]
        public async Task GetFeedbackImageList_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/feedback/images/bottle/anomaly");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task GetFeedbackImageList_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/images/bottle/anomaly");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task GetFeedbackImageList_Returns400_WhenLabelIsInvalid()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/images/bottle/invalid_label");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task GetFeedbackImageList_Returns200_WithFileNames_AfterFeedbackSubmitted()
        {
            // Arrange
            var (userToken, _) = await RegisterAndLoginAsync("User");
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");

            var userClient = CreateAuthenticatedClient(userToken);
            var adminClient = CreateAuthenticatedClient(adminToken);

            await userClient.PostAsync("/api/v1/feedback",
                CreateFeedbackFormContent("cable", true));

            // Act
            var response = await adminClient.GetAsync("/api/v1/feedback/images/cable/anomaly");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            json.GetArrayLength().Should().BeGreaterThanOrEqualTo(1);
        }
        #endregion

        #region GetFeedbackImage Tests
        [Fact]
        public async Task GetFeedbackImage_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/feedback/images/bottle/anomaly/test.png");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task GetFeedbackImage_Returns400_WhenLabelIsInvalid()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/images/bottle/wrong_label/test.png");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task GetFeedbackImage_Returns400_WhenFilenameContainsPathTraversal()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/images/bottle/anomaly/..%2Fsecrets.txt");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task GetFeedbackImage_Returns404_WhenFileDoesNotExist()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/feedback/images/bottle/anomaly/nonexistent.png");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.NotFound);
        }

        [Fact]
        public async Task GetFeedbackImage_Returns200_WithImageBytes_WhenFileExists()
        {
            // Arrange
            var (userToken, _) = await RegisterAndLoginAsync("User");
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");

            var userClient = CreateAuthenticatedClient(userToken);
            var adminClient = CreateAuthenticatedClient(adminToken);

            await userClient.PostAsync("/api/v1/feedback",
                CreateFeedbackFormContent("hazelnut", true, "photo.png"));

            var listResponse = await adminClient.GetAsync("/api/v1/feedback/images/hazelnut/anomaly");
            var listBody = await listResponse.Content.ReadAsStringAsync();
            var files = JsonSerializer.Deserialize<JsonElement>(listBody);
            var fileName = files[0].GetString()!;

            // Act
            var response = await adminClient.GetAsync($"/api/v1/feedback/images/hazelnut/anomaly/{fileName}");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            response.Content.Headers.ContentType!.MediaType.Should().Be("image/png");
        }
        #endregion
    }
}