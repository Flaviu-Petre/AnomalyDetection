using AnomalyDetection.Api.Models.Domain;
using AnomalyDetection.Api.Models.DTOs;
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
using AnomalyDetection.Api.Data;

namespace AnomalyDetection.Tests.Integration
{
    public class ModelsTestFactory : WebApplicationFactory<Program>
    {
        private readonly string _dbName = "ModelsTestDb_" + Guid.NewGuid();

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
                services.AddSingleton<IRouterService>(_ => Mock.Of<IRouterService>());

                services.AddDbContext<AppDbContext>(options =>
                    options.UseInMemoryDatabase(_dbName));
            });
        }
    }

    public class ModelsIntegrationTests : IClassFixture<ModelsTestFactory>, IDisposable
    {
        #region Setup
        private readonly ModelsTestFactory _factory;
        private readonly HttpClient _client;

        public ModelsIntegrationTests(ModelsTestFactory factory)
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

        private static List<ModelInfo> CreateFakeModelList() => new List<ModelInfo>
        {
            new ModelInfo { Category = "bottle",  Threshold = 0.5f },
            new ModelInfo { Category = "capsule", Threshold = 0.6f },
            new ModelInfo { Category = "cable",   Threshold = 0.4f }
        };
        #endregion

        #region GetAllModels Tests
        [Fact]
        public async Task GetAllModels_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/models/get_all_models");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task GetAllModels_Returns200_WhenUserIsAuthenticated()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.GetAvailableModels())
                .Returns(new List<ModelInfo>());

            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/models/get_all_models");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetAllModels_Returns200_WhenUserIsAdmin()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.GetAvailableModels())
                .Returns(CreateFakeModelList());

            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/models/get_all_models");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetAllModels_ReturnsCorrectModelList()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.GetAvailableModels())
                .Returns(CreateFakeModelList());

            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/models/get_all_models");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetArrayLength().Should().Be(3);
            var categories = json.EnumerateArray()
                .Select(m => m.GetProperty("category").GetString())
                .ToList();
            categories.Should().Contain("bottle");
            categories.Should().Contain("capsule");
            categories.Should().Contain("cable");
        }

        [Fact]
        public async Task GetAllModels_ReturnsEmptyList_WhenNoModelsAvailable()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.GetAvailableModels())
                .Returns(new List<ModelInfo>());

            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/models/get_all_models");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetArrayLength().Should().Be(0);
        }
        #endregion

        #region DeleteModel Tests
        [Fact]
        public async Task DeleteModel_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.DeleteAsync("/api/v1/models/delete_category?category=bottle");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task DeleteModel_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.DeleteAsync("/api/v1/models/delete_category?category=bottle");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task DeleteModel_Returns200_WhenAdminDeletesExistingCategory()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.DeleteModel("bottle"));

            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.DeleteAsync("/api/v1/models/delete_category?category=bottle");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task DeleteModel_CallsService_WithCorrectCategory()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.DeleteModel("capsule"));

            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            await client.DeleteAsync("/api/v1/models/delete_category?category=capsule");

            // Assert
            _factory.ModelManagerMock.Verify(m => m.DeleteModel("capsule"), Times.AtLeastOnce);
        }

        [Fact]
        public async Task DeleteModel_Returns400_WhenCategoryIsEmpty()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.DeleteAsync("/api/v1/models/delete_category?category=");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task DeleteModel_Returns404_WhenCategoryDoesNotExist()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.DeleteModel("nonexistent"))
                .Throws(new FileNotFoundException("Model not found"));

            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.DeleteAsync("/api/v1/models/delete_category?category=nonexistent");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.NotFound);
        }

        [Fact]
        public async Task DeleteModel_Returns400_WhenCategoryContainsPathTraversal()
        {
            // Arrange
            _factory.ModelManagerMock
                .Setup(m => m.DeleteModel(It.IsAny<string>()))
                .Throws(new ArgumentException("Invalid category name — path traversal not allowed."));

            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.DeleteAsync("/api/v1/models/delete_category?category=../malicious");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }
        #endregion

        #region UploadModel Tests
        [Fact]
        public async Task UploadModel_Returns401_WhenNoTokenProvided()
        {
            // Arrange
            var content = new MultipartFormDataContent();
            content.Add(new StringContent("bottle"), "category");

            // Act
            var response = await _client.PostAsync("/api/v1/models/upload_model", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task UploadModel_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            var content = new MultipartFormDataContent();
            content.Add(new StringContent("bottle"), "category");

            // Act
            var response = await client.PostAsync("/api/v1/models/upload_model", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task UploadModel_Returns400_WhenCategoryIsMissing()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            var content = new MultipartFormDataContent();
            content.Add(new StringContent(""), "category");
            content.Add(new ByteArrayContent(new byte[] { 1, 2, 3 })
            {
                Headers = { ContentDisposition = new ContentDispositionHeaderValue("form-data")
                    { Name = "bankFile", FileName = "model.npz" } }
            });
            content.Add(new ByteArrayContent(new byte[] { 1, 2, 3 })
            {
                Headers = { ContentDisposition = new ContentDispositionHeaderValue("form-data")
                    { Name = "jsonMetadata", FileName = "meta.json" } }
            });

            // Act
            var response = await client.PostAsync("/api/v1/models/upload_model", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task UploadModel_Returns400_WhenBankFileHasWrongExtension()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            var content = new MultipartFormDataContent();
            content.Add(new StringContent("bottle"), "category");
            content.Add(new ByteArrayContent(new byte[] { 1, 2, 3 })
            {
                Headers = { ContentDisposition = new ContentDispositionHeaderValue("form-data")
                    { Name = "bankFile", FileName = "model.zip" } }
            });
            content.Add(new ByteArrayContent(new byte[] { 1, 2, 3 })
            {
                Headers = { ContentDisposition = new ContentDispositionHeaderValue("form-data")
                    { Name = "jsonMetadata", FileName = "meta.json" } }
            });

            // Act
            var response = await client.PostAsync("/api/v1/models/upload_model", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task UploadModel_Returns400_WhenMetadataFileHasWrongExtension()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            var content = new MultipartFormDataContent();
            content.Add(new StringContent("bottle"), "category");
            content.Add(new ByteArrayContent(new byte[] { 1, 2, 3 })
            {
                Headers = { ContentDisposition = new ContentDispositionHeaderValue("form-data")
                    { Name = "bankFile", FileName = "model.npz" } }
            });
            content.Add(new ByteArrayContent(new byte[] { 1, 2, 3 })
            {
                Headers = { ContentDisposition = new ContentDispositionHeaderValue("form-data")
                    { Name = "jsonMetadata", FileName = "meta.txt" } }
            });

            // Act
            var response = await client.PostAsync("/api/v1/models/upload_model", content);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }
        #endregion
    }
}