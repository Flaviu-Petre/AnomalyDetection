using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Moq;
using System.Net;
using System.Text;
using System.Text.Json;

namespace AnomalyDetection.Tests.Integration
{
    public class AuthTestFactory : WebApplicationFactory<Program>
    {
        private readonly string _dbName = "AuthTestDb_" + Guid.NewGuid();

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

    public class AuthIntegrationTests : IClassFixture<AuthTestFactory>, IDisposable
    {
        #region Setup
        private readonly AuthTestFactory _factory;
        private readonly HttpClient _client;

        public AuthIntegrationTests(AuthTestFactory factory)
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

        private async Task<(string Username, string Password, string Email)> RegisterUserAsync(
            string? username = null, string? password = null, string? email = null)
        {
            username ??= "user_" + Guid.NewGuid().ToString("N")[..8];
            password ??= "Password123!";
            email ??= username + "@test.com";

            await _client.PostAsync("/api/v1/auth/register", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            return (username, password, email);
        }

        private async Task<string> GetTokenAsync(string role = "User")
        {
            var (username, password, email) = await RegisterUserAsync();

            if (role == "Admin")
            {
                using var scope = _factory.Services.CreateScope();
                var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                var u = db.Users.First(x => x.Username == username);
                u.Role = "Admin";
                db.SaveChanges();
            }

            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);
            return json.GetProperty("token").GetString()!;
        }
        private async Task<string> GetResetTokenAsync(string email)
        {
            var response = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = email
            }));

            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);
            return json.GetProperty("resetToken").GetString()!;
        }
        #endregion

        #region Register Tests
        [Fact]
        public async Task Register_Returns200_WhenRequestIsValid()
        {
            // Arrange
            var request = new
            {
                Username = "user_" + Guid.NewGuid().ToString("N")[..8],
                Password = "Password123!",
                Email = "valid_" + Guid.NewGuid().ToString("N")[..8] + "@test.com"
            };

            // Act
            var response = await _client.PostAsync("/api/v1/auth/register", JsonContent(request));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task Register_Returns400_WhenUsernameIsAlreadyTaken()
        {
            // Arrange
            var username = "dup_" + Guid.NewGuid().ToString("N")[..8];
            var request = new
            {
                Username = username,
                Password = "Password123!",
                Email = username + "@test.com"
            };

            await _client.PostAsync("/api/v1/auth/register", JsonContent(request));

            // Act
            var response = await _client.PostAsync("/api/v1/auth/register", JsonContent(request));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("Username already exists");
        }

        [Fact]
        public async Task Register_Returns400_WhenEmailIsInvalid()
        {
            // Arrange
            var request = new
            {
                Username = "user_" + Guid.NewGuid().ToString("N")[..8],
                Password = "Password123!",
                Email = "not-a-valid-email"
            };

            // Act
            var response = await _client.PostAsync("/api/v1/auth/register", JsonContent(request));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task Register_Returns400_WhenEmailIsEmpty()
        {
            // Arrange
            var request = new
            {
                Username = "user_" + Guid.NewGuid().ToString("N")[..8],
                Password = "Password123!",
                Email = ""
            };

            // Act
            var response = await _client.PostAsync("/api/v1/auth/register", JsonContent(request));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task Register_StoresUser_InDatabase()
        {
            // Arrange
            var username = "dbuser_" + Guid.NewGuid().ToString("N")[..8];
            var email = username + "@test.com";

            // Act
            await _client.PostAsync("/api/v1/auth/register", JsonContent(new
            {
                Username = username,
                Password = "Password123!",
                Email = email
            }));

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = db.Users.FirstOrDefault(u => u.Username == username);

            user.Should().NotBeNull();
            user!.Email.Should().Be(email);
            user.Role.Should().Be("User");
            user.PasswordHash.Should().StartWith("$2");
        }

        [Fact]
        public async Task Register_DoesNotStorePasswordInPlainText()
        {
            // Arrange
            var username = "secure_" + Guid.NewGuid().ToString("N")[..8];

            // Act
            await _client.PostAsync("/api/v1/auth/register", JsonContent(new
            {
                Username = username,
                Password = "MyPlainPassword!",
                Email = username + "@test.com"
            }));

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = db.Users.FirstOrDefault(u => u.Username == username);

            user.Should().NotBeNull();
            user!.PasswordHash.Should().NotBe("MyPlainPassword!");
            user.PasswordHash.Should().StartWith("$2");
        }
        #endregion

        #region Login Tests
        [Fact]
        public async Task Login_Returns200_WithToken_WhenCredentialsAreValid()
        {
            // Arrange
            var (username, password, email) = await RegisterUserAsync();

            // Act
            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            var body = await response.Content.ReadAsStringAsync();
            body.Should().Contain("token");
            body.Should().Contain("role");
        }

        [Fact]
        public async Task Login_Returns401_WhenPasswordIsWrong()
        {
            // Arrange
            var (username, _, email) = await RegisterUserAsync();

            // Act
            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = "WrongPassword!",
                Email = email
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task Login_Returns401_WhenEmailDoesNotMatch()
        {
            // Arrange
            var (username, password, _) = await RegisterUserAsync();

            // Act
            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = "wrong@test.com"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task Login_Returns401_WhenUserDoesNotExist()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = "nonexistent_" + Guid.NewGuid().ToString("N")[..8],
                Password = "Password123!",
                Email = "nonexistent@test.com"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task Login_ReturnsJwtToken_WithCorrectRole()
        {
            // Arrange
            var (username, password, email) = await RegisterUserAsync();

            // Act
            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            json.GetProperty("role").GetString().Should().Be("User");
            json.GetProperty("token").GetString().Should().NotBeNullOrEmpty();
        }

        [Fact]
        public async Task Login_Returns401_WhenUsernameIsNull()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = (string?)null,
                Password = "Password123!",
                Email = "test@test.com"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }
        #endregion

        #region Full Flow Tests
        [Fact]
        public async Task RegisterThenLogin_FullFlow_ReturnsValidToken()
        {
            // Arrange
            var username = "flow_" + Guid.NewGuid().ToString("N")[..8];
            var password = "FlowPassword123!";
            var email = username + "@test.com";

            // Act 
            var registerResponse = await _client.PostAsync("/api/v1/auth/register", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            // Act
            var loginResponse = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = password,
                Email = email
            }));

            var body = await loginResponse.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            registerResponse.StatusCode.Should().Be(HttpStatusCode.OK);
            loginResponse.StatusCode.Should().Be(HttpStatusCode.OK);
            json.GetProperty("token").GetString().Should().NotBeNullOrEmpty();
        }

        [Fact]
        public async Task ProtectedEndpoint_Returns401_WhenNoTokenProvided()
        {
            // Act
            var response = await _client.GetAsync("/api/v1/users");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        [Fact]
        public async Task ProtectedEndpoint_Returns200_WhenValidAdminTokenProvided()
        {
            // Arrange 
            var token = await GetTokenAsync(role: "Admin");

            // Act
            var client = _factory.CreateClient();
            client.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

            var response = await client.GetAsync("/api/v1/users");

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }
        #endregion

        #region ForgotPassword Tests
        [Fact]
        public async Task ForgotPassword_Returns200_WhenEmailExists()
        {
            // Arrange
            var (_, _, email) = await RegisterUserAsync();

            // Act
            var response = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = email
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task ForgotPassword_Returns200_WhenEmailDoesNotExist()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = "nonexistent_" + Guid.NewGuid().ToString("N")[..8] + "@test.com"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task ForgotPassword_Returns400_WhenEmailIsEmpty()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = ""
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task ForgotPassword_ReturnsNullToken_WhenEmailDoesNotExist()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = "nonexistent_" + Guid.NewGuid().ToString("N")[..8] + "@test.com"
            }));

            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            json.GetProperty("resetToken").GetString().Should().BeNull();
        }

        [Fact]
        public async Task ForgotPassword_ReturnsToken_WhenEmailExists()
        {
            // Arrange
            var (_, _, email) = await RegisterUserAsync();

            // Act
            var response = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = email
            }));

            var body = await response.Content.ReadAsStringAsync();
            var json = JsonSerializer.Deserialize<JsonElement>(body);

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            json.GetProperty("resetToken").GetString().Should().NotBeNullOrEmpty();
        }

        [Fact]
        public async Task ForgotPassword_SetsTokenInDatabase_WhenEmailExists()
        {
            // Arrange
            var (username, _, email) = await RegisterUserAsync();

            // Act
            await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = email
            }));

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = db.Users.FirstOrDefault(u => u.Username == username);

            user.Should().NotBeNull();
            user!.PasswordResetToken.Should().NotBeNullOrEmpty();
            user.PasswordResetTokenExpiry.Should().NotBeNull();
            user.PasswordResetTokenExpiry.Should().BeAfter(DateTime.UtcNow);
        }
        #endregion

        #region ResetPassword Tests
        [Fact]
        public async Task ResetPassword_Returns200_WhenTokenAndPasswordAreValid()
        {
            // Arrange
            var (_, _, email) = await RegisterUserAsync();
            var token = await GetResetTokenAsync(email);

            // Act
            var response = await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = "NewPassword456!"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }

        [Fact]
        public async Task ResetPassword_Returns400_WhenTokenIsInvalid()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = "completely-invalid-token-that-does-not-exist",
                NewPassword = "NewPassword456!"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task ResetPassword_Returns400_WhenTokenIsEmpty()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = "",
                NewPassword = "NewPassword456!"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task ResetPassword_Returns400_WhenPasswordIsEmpty()
        {
            // Act
            var response = await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = "some-token",
                NewPassword = ""
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }

        [Fact]
        public async Task ResetPassword_UpdatesPasswordHash_InDatabase()
        {
            // Arrange
            var (username, _, email) = await RegisterUserAsync();
            var token = await GetResetTokenAsync(email);

            using var scopeBefore = _factory.Services.CreateScope();
            var dbBefore = scopeBefore.ServiceProvider.GetRequiredService<AppDbContext>();
            var oldHash = dbBefore.Users.First(u => u.Username == username).PasswordHash;

            // Act
            await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = "NewPassword456!"
            }));

            // Assert
            using var scopeAfter = _factory.Services.CreateScope();
            var dbAfter = scopeAfter.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = dbAfter.Users.First(u => u.Username == username);

            user.PasswordHash.Should().NotBe(oldHash);
            user.PasswordHash.Should().StartWith("$2");
        }

        [Fact]
        public async Task ResetPassword_ClearsTokenFromDatabase_AfterSuccessfulReset()
        {
            // Arrange
            var (username, _, email) = await RegisterUserAsync();
            var token = await GetResetTokenAsync(email);

            // Act
            await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = "NewPassword456!"
            }));

            // Assert
            using var scope = _factory.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var user = db.Users.First(u => u.Username == username);

            user.PasswordResetToken.Should().BeNull();
            user.PasswordResetTokenExpiry.Should().BeNull();
        }

        [Fact]
        public async Task ResetPassword_TokenCannotBeReusedAfterSuccessfulReset()
        {
            // Arrange
            var (_, _, email) = await RegisterUserAsync();
            var token = await GetResetTokenAsync(email);

            await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = "NewPassword456!"
            }));

            // Act
            var response = await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = "AnotherPassword789!"
            }));

            // Assert
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        }
        #endregion

        #region Full Reset Flow Tests
        [Fact]
        public async Task ForgotPasswordThenResetThenLogin_FullFlow_Succeeds()
        {
            // Arrange
            var (username, _, email) = await RegisterUserAsync();
            var newPassword = "BrandNewPassword123!";

            // Act
            var forgotResponse = await _client.PostAsync("/api/v1/auth/forgot-password", JsonContent(new
            {
                Email = email
            }));
            var forgotBody = await forgotResponse.Content.ReadAsStringAsync();
            var forgotJson = JsonSerializer.Deserialize<JsonElement>(forgotBody);
            var token = forgotJson.GetProperty("resetToken").GetString()!;

            var resetResponse = await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = newPassword
            }));

            var loginResponse = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = newPassword,
                Email = email
            }));

            var loginBody = await loginResponse.Content.ReadAsStringAsync();
            var loginJson = JsonSerializer.Deserialize<JsonElement>(loginBody);

            // Assert
            forgotResponse.StatusCode.Should().Be(HttpStatusCode.OK);
            resetResponse.StatusCode.Should().Be(HttpStatusCode.OK);
            loginResponse.StatusCode.Should().Be(HttpStatusCode.OK);
            loginJson.GetProperty("token").GetString().Should().NotBeNullOrEmpty();
        }

        [Fact]
        public async Task OldPasswordNoLongerWorksAfterReset()
        {
            // Arrange
            var (username, oldPassword, email) = await RegisterUserAsync();
            var token = await GetResetTokenAsync(email);

            await _client.PostAsync("/api/v1/auth/reset-password", JsonContent(new
            {
                Token = token,
                NewPassword = "BrandNewPassword123!"
            }));

            // Act
            var loginResponse = await _client.PostAsync("/api/v1/auth/login", JsonContent(new
            {
                Username = username,
                Password = oldPassword,
                Email = email
            }));

            // Assert
            loginResponse.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }
        #endregion
    }
}