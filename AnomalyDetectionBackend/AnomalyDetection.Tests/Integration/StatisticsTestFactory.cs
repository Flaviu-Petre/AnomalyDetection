using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.Entities;
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
    public class StatisticsTestFactory : WebApplicationFactory<Program>
    {
        private readonly string _dbName = "StatsTestDb_" + Guid.NewGuid();

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

    public class StatisticsIntegrationTests : IClassFixture<StatisticsTestFactory>, IDisposable
    {
        #region Setup
        private readonly StatisticsTestFactory _factory;
        private readonly HttpClient _client;

        public StatisticsIntegrationTests(StatisticsTestFactory factory)
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

        private void SeedInferenceRecord(int userId, string category, bool isAnomaly, float score, DateTime? timestamp = null)
        {
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            db.InferenceRecords.Add(new InferenceRecord
            {
                UserId = userId,
                Category = category,
                IsAnomaly = isAnomaly,
                Score = score,
                ThresholdUsed = 0.5f,
                Timestamp = timestamp ?? DateTime.UtcNow.AddDays(-1),
                ImageName = "test.png"
            });
            db.SaveChanges();
        }

        private async Task<string> GetUsernameFromToken(string token)
        {
            var parts = token.Split('.');
            var payload = parts[1];
            var padded = payload.PadRight(payload.Length + (4 - payload.Length % 4) % 4, '=');
            var json = JsonSerializer.Deserialize<JsonElement>(
                System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(padded)));
            return json.GetProperty("sub").GetString()!;
        }

        private static int GetUserIdFromToken(string token)
        {
            var parts = token.Split('.');
            var payload = parts[1];
            var padded = payload.PadRight(payload.Length + (4 - payload.Length % 4) % 4, '=');
            var json = JsonSerializer.Deserialize<JsonElement>(
                System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(padded)));
            return int.Parse(json.GetProperty("id").GetString()!);
        }
        #endregion

        #region Dashboard Stats Tests — Unauthorized
        [Fact]
        public async Task GetDashboardStats_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/statistics");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }
        #endregion

        #region Dashboard Stats Tests — User
        [Fact]
        public async Task GetDashboardStats_Returns200_WhenUserIsAuthenticated()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/statistics");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetDashboardStats_ReturnsZeroStats_WhenNoInferencesExist()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/statistics");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalInferencesThisWeek").GetInt32().Should().Be(0);
            json.GetProperty("totalAnomaliesThisWeek").GetInt32().Should().Be(0);
            json.GetProperty("overallAnomalyRatePercentage").GetDouble().Should().Be(0);
        }

        [Fact]
        public async Task GetDashboardStats_ReturnsOnlyOwnStats_ForNonAdminUser()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId + 99, "capsule", false, 0.2f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalInferencesThisWeek").GetInt32().Should().Be(1);
        }

        [Fact]
        public async Task GetDashboardStats_ReturnsAllStats_ForAdmin()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId + 99, "capsule", false, 0.2f);
            SeedInferenceRecord(userId + 98, "cable", true, 0.8f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalInferencesThisWeek").GetInt32().Should().BeGreaterThanOrEqualTo(3);
        }

        [Fact]
        public async Task GetDashboardStats_ReturnsCorrectAnomalyRate()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId, "bottle", false, 0.1f);
            SeedInferenceRecord(userId, "bottle", false, 0.2f);
            SeedInferenceRecord(userId, "bottle", false, 0.3f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("overallAnomalyRatePercentage").GetDouble().Should().BeGreaterThan(0);
        }

        [Fact]
        public async Task GetDashboardStats_DoesNotReturnOldRecords_OlderThan7Days()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f, DateTime.UtcNow.AddDays(-10));
            SeedInferenceRecord(userId, "capsule", false, 0.2f, DateTime.UtcNow.AddDays(-1));

            // Act
            var response = await client.GetAsync("/api/v1/statistics");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalInferencesThisWeek").GetInt32().Should().Be(1);
        }
        #endregion

        #region History Tests
        [Fact]
        public async Task GetHistory_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/statistics/history");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task GetHistory_Returns200_WhenUserIsAuthenticated()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetHistory_ReturnsPaginatedResult()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?page=1&pageSize=10");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.TryGetProperty("items", out _).Should().BeTrue();
            json.TryGetProperty("totalCount", out _).Should().BeTrue();
            json.TryGetProperty("pageNumber", out _).Should().BeTrue();
            json.TryGetProperty("pageSize", out _).Should().BeTrue();
        }

        [Fact]
        public async Task GetHistory_ReturnsOnlyOwnRecords_ForNonAdminUser()
        {
            // Arrange
            var token = await GetTokenAsync("User");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId + 99, "capsule", false, 0.2f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalCount").GetInt32().Should().Be(1);
        }

        [Fact]
        public async Task GetHistory_ReturnsAllRecords_ForAdmin()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId + 99, "capsule", false, 0.2f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalCount").GetInt32().Should().BeGreaterThanOrEqualTo(2);
        }

        [Fact]
        public async Task GetHistory_RespectsPageSize()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            for (int i = 0; i < 5; i++)
                SeedInferenceRecord(userId, "bottle", i % 2 == 0, 0.5f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?page=1&pageSize=2");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("items").GetArrayLength().Should().BeLessThanOrEqualTo(2);
            json.GetProperty("pageSize").GetInt32().Should().Be(2);
        }
        #endregion

        #region History Filter Tests
        [Fact]
        public async Task GetHistory_FiltersByIsAnomaly_ReturnsOnlyAnomalies()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId, "capsule", false, 0.2f);
            SeedInferenceRecord(userId, "cable", true, 0.8f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?isAnomaly=true");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var items = json.GetProperty("items").EnumerateArray().ToList();
            items.Should().NotBeEmpty();
            items.All(i => i.GetProperty("isAnomaly").GetBoolean()).Should().BeTrue();
        }

        [Fact]
        public async Task GetHistory_FiltersByIsAnomaly_ReturnsOnlyNormal()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId, "capsule", false, 0.2f);
            SeedInferenceRecord(userId, "cable", false, 0.1f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?isAnomaly=false");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var items = json.GetProperty("items").EnumerateArray().ToList();
            items.Should().NotBeEmpty();
            items.All(i => !i.GetProperty("isAnomaly").GetBoolean()).Should().BeTrue();
        }

        [Fact]
        public async Task GetHistory_FiltersByCategory_ReturnsOnlyMatchingCategory()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId, "capsule", false, 0.2f);
            SeedInferenceRecord(userId, "bottle", false, 0.3f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?category=bottle");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var items = json.GetProperty("items").EnumerateArray().ToList();
            items.Should().NotBeEmpty();
            items.All(i => i.GetProperty("category").GetString() == "bottle").Should().BeTrue();
        }

        [Fact]
        public async Task GetHistory_FiltersByCategory_ReturnsEmptyResult_WhenNoneMatch()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?category=zipper");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("totalCount").GetInt32().Should().Be(0);
            json.GetProperty("items").GetArrayLength().Should().Be(0);
        }

        [Fact]
        public async Task GetHistory_FiltersByUsername_ReturnsOnlyMatchingUser_ForAdmin()
        {
            // Arrange
            var adminToken = await GetTokenAsync("Admin");
            var adminUserId = GetUserIdFromToken(adminToken);
            var adminClient = CreateAuthenticatedClient(adminToken);

            var userToken = await GetTokenAsync("User");
            var userUserId = GetUserIdFromToken(userToken);
            var username = await GetUsernameFromToken(userToken);

            SeedInferenceRecord(userUserId, "bottle", true, 0.9f);
            SeedInferenceRecord(adminUserId, "capsule", false, 0.2f);

            // Act
            var response = await adminClient.GetAsync($"/api/v1/statistics/history?filterUsername={username}");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var items = json.GetProperty("items").EnumerateArray().ToList();
            items.Should().NotBeEmpty();
            items.All(i => i.GetProperty("username").GetString() == username).Should().BeTrue();
        }

        [Fact]
        public async Task GetHistory_IgnoresUsernameFilter_ForNonAdmin()
        {
            // Arrange
            var userToken = await GetTokenAsync("User");
            var userId = GetUserIdFromToken(userToken);
            var userClient = CreateAuthenticatedClient(userToken);

            var otherToken = await GetTokenAsync("User");
            var otherUserId = GetUserIdFromToken(otherToken);
            var otherUsername = await GetUsernameFromToken(otherToken);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(otherUserId, "capsule", false, 0.2f);

            // Act
            var response = await userClient.GetAsync($"/api/v1/statistics/history?filterUsername={otherUsername}");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var items = json.GetProperty("items").EnumerateArray().ToList();
            items.All(i => i.GetProperty("username").GetString() != otherUsername).Should().BeTrue();
        }

        [Fact]
        public async Task GetHistory_CombinesFilters_IsAnomalyAndCategory()
        {
            // Arrange
            var token = await GetTokenAsync("Admin");
            var userId = GetUserIdFromToken(token);
            var client = CreateAuthenticatedClient(token);

            SeedInferenceRecord(userId, "bottle", true, 0.9f);
            SeedInferenceRecord(userId, "bottle", false, 0.2f);
            SeedInferenceRecord(userId, "capsule", true, 0.8f);

            // Act
            var response = await client.GetAsync("/api/v1/statistics/history?category=bottle&isAnomaly=true");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            var items = json.GetProperty("items").EnumerateArray().ToList();
            items.Should().NotBeEmpty();
            items.All(i =>
                i.GetProperty("category").GetString() == "bottle" &&
                i.GetProperty("isAnomaly").GetBoolean()
            ).Should().BeTrue();
        }
        #endregion
    }
}