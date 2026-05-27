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
    public class UsersTestFactory : WebApplicationFactory<Program>
    {
        private readonly string _dbName = "UsersTestDb_" + Guid.NewGuid();

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

    public class UsersIntegrationTests : IClassFixture<UsersTestFactory>, IDisposable
    {
        #region Setup
        private readonly UsersTestFactory _factory;
        private readonly HttpClient _client;

        public UsersIntegrationTests(UsersTestFactory factory)
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
        #endregion

        #region GetAllUsers Tests
        [Fact]
        public async Task GetAllUsers_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/users");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task GetAllUsers_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/users");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task GetAllUsers_Returns200_WhenUserIsAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/users");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task GetAllUsers_ReturnsListOfUsers_WhenAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            await RegisterAndLoginAsync("User");

            // Act
            var response = await client.GetAsync("/api/v1/users");
            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetArrayLength().Should().BeGreaterThanOrEqualTo(2);
        }

        [Fact]
        public async Task GetAllUsers_DoesNotReturnPasswordHash()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.GetAsync("/api/v1/users");
            var body = await response.Content.ReadAsStringAsync();

            // Assert
            body.Should().NotContain("$2");
        }
        #endregion

        #region UpdateRole Tests
        [Fact]
        public async Task UpdateRole_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.PutAsync("/api/v1/users/1/role",
                JsonContent(new { Role = "Admin" }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task UpdateRole_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.PutAsync("/api/v1/users/1/role",
                JsonContent(new { Role = "Admin" }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task UpdateRole_Returns200_WhenAdminChangesAnotherUsersRole()
        {
            // Arrange
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var (_, targetUserId) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            var response = await client.PutAsync($"/api/v1/users/{targetUserId}/role",
                JsonContent(new { Role = "Admin" }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task UpdateRole_ActuallyChangesRole_InDatabase()
        {
            // Arrange
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var (_, targetUserId) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            await client.PutAsync($"/api/v1/users/{targetUserId}/role",
                JsonContent(new { Role = "Admin" }));

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = db.Users.First(u => u.Id == targetUserId);
            user.Role.Should().Be("Admin");
        }

        [Fact]
        public async Task UpdateRole_Returns400_WhenAdminTriesToChangeOwnRole()
        {
            // Arrange
            var (adminToken, adminId) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            var response = await client.PutAsync($"/api/v1/users/{adminId}/role",
                JsonContent(new { Role = "User" }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("Security Policy");
        }

        [Fact]
        public async Task UpdateRole_Returns400_WhenRoleIsInvalid()
        {
            // Arrange
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var (_, targetUserId) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            var response = await client.PutAsync($"/api/v1/users/{targetUserId}/role",
                JsonContent(new { Role = "SuperAdmin" }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("Invalid role");
        }
        #endregion

        #region DeleteUser Tests
        [Fact]
        public async Task DeleteUser_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.DeleteAsync("/api/v1/users/1");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task DeleteUser_Returns403_WhenUserIsNotAdmin()
        {
            // Arrange
            var (token, _) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(token);

            // Act
            var response = await client.DeleteAsync("/api/v1/users/1");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Forbidden);
        }

        [Fact]
        public async Task DeleteUser_Returns200_WhenAdminDeletesAnotherUser()
        {
            // Arrange
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var (_, targetUserId) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            var response = await client.DeleteAsync($"/api/v1/users/{targetUserId}");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task DeleteUser_ActuallyRemovesUser_FromDatabase()
        {
            // Arrange
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var (_, targetUserId) = await RegisterAndLoginAsync("User");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            await client.DeleteAsync($"/api/v1/users/{targetUserId}");

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = db.Users.FirstOrDefault(u => u.Id == targetUserId);
            user.Should().BeNull();
        }

        [Fact]
        public async Task DeleteUser_Returns400_WhenAdminTriesToDeleteOwnAccount()
        {
            // Arrange
            var (adminToken, adminId) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            var response = await client.DeleteAsync($"/api/v1/users/{adminId}");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("Security Policy");
        }

        [Fact]
        public async Task DeleteUser_Returns404_WhenUserDoesNotExist()
        {
            // Arrange
            var (adminToken, _) = await RegisterAndLoginAsync("Admin");
            var client = CreateAuthenticatedClient(adminToken);

            // Act
            var response = await client.DeleteAsync("/api/v1/users/99999");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.NotFound);
        }
        #endregion
    }
}