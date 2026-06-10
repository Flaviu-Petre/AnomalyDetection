using AnomalyDetection.Api.Models.Configuration;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services;
using FluentAssertions;
using Microsoft.Extensions.Options;
using Moq;

namespace AnomalyDetection.Tests.Unit.Services
{
    public class AuthServiceTests
    {
        #region Setup
        private readonly Mock<IUserRepository> _mockUserRepo;
        private readonly IOptions<JwtSettings> _jwtOptions;
        private readonly AuthService _authService;

        public AuthServiceTests()
        {
            _mockUserRepo = new Mock<IUserRepository>();

            _jwtOptions = Options.Create(new JwtSettings
            {
                Secret = "super-secret-key-for-testing-minimum-32-chars!!",
                Issuer = "TestIssuer",
                Audience = "TestAudience",
                ExpirationHours = 8
            });

            _authService = new AuthService(_mockUserRepo.Object, _jwtOptions);
        }
        #endregion

        #region IsUsernameTaken Tests
        [Fact]
        public void IsUsernameTaken_ReturnsTrue_WhenUserExists()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.UserExists("name")).Returns(true);

            // Act
            var result = _authService.IsUsernameTaken("name");

            // Assert
            result.Should().BeTrue();
            _mockUserRepo.Verify(r => r.UserExists("name"), Times.Once);
        }

        [Fact]
        public void IsUsernameTaken_ReturnsFalse_WhenUserDoesNotExist()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.UserExists("newuser")).Returns(false);

            // Act
            var result = _authService.IsUsernameTaken("newuser");

            // Assert
            result.Should().BeFalse();
        }
        #endregion

        #region RegisterUser Tests
        [Fact]
        public void RegisterUser_CallsAddUser_WhenDataIsValid()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.AddUser(It.IsAny<User>()));

            // Act
            _authService.RegisterUser("name", "Password123!", "name@test.com");

            // Assert
            _mockUserRepo.Verify(r => r.AddUser(It.Is<User>(u =>
                u.Username == "name" &&
                u.Email == "name@test.com" &&
                u.Role == "User" &&
                !string.IsNullOrEmpty(u.PasswordHash)
            )), Times.Once);
        }

        [Fact]
        public void RegisterUser_HashesPassword_BeforeSaving()
        {
            // Arrange
            User? savedUser = null;
            _mockUserRepo.Setup(r => r.AddUser(It.IsAny<User>()))
                         .Callback<User>(u => savedUser = u);

            // Act
            _authService.RegisterUser("name", "Password123!", "name@test.com");

            // Assert
            savedUser.Should().NotBeNull();
            savedUser!.PasswordHash.Should().NotBe("Password123!");
            savedUser.PasswordHash.Should().StartWith("$2");
        }

        [Fact]
        public void RegisterUser_ThrowsArgumentException_WhenEmailIsEmpty()
        {
            // Act
            Action act = () => _authService.RegisterUser("name", "Password123!", "");

            // Assert
            act.Should().Throw<ArgumentException>()
               .WithMessage("Email address is required.");
        }

        [Fact]
        public void RegisterUser_ThrowsArgumentException_WhenEmailIsWhitespace()
        {
            // Act
            Action act = () => _authService.RegisterUser("name", "Password123!", "   ");

            // Assert
            act.Should().Throw<ArgumentException>()
               .WithMessage("Email address is required.");
        }

        [Fact]
        public void RegisterUser_ThrowsArgumentException_WhenEmailIsInvalid()
        {
            // Act
            Action act = () => _authService.RegisterUser("name", "Password123!", "not-an-email");

            // Assert
            act.Should().Throw<ArgumentException>()
               .WithMessage("Invalid email address format.");
        }

        [Fact]
        public void RegisterUser_DoesNotCallAddUser_WhenEmailIsInvalid()
        {
            // Act
            try { _authService.RegisterUser("name", "Password123!", "bad-email"); } catch { }

            // Assert
            _mockUserRepo.Verify(r => r.AddUser(It.IsAny<User>()), Times.Never);
        }
        #endregion

        #region Login Tests
        [Fact]
        public void Login_ReturnsNull_WhenUsernameIsNull()
        {
            // Act
            var result = _authService.Login(null, "name@test.com", "Password123!");

            // Assert
            result.Should().BeNull();
        }

        [Fact]
        public void Login_ReturnsNull_WhenEmailIsNull()
        {
            // Act
            var result = _authService.Login("name", null, "Password123!");

            // Assert
            result.Should().BeNull();
        }

        [Fact]
        public void Login_ReturnsNull_WhenUsernameIsEmpty()
        {
            // Act
            var result = _authService.Login("", "name@test.com", "Password123!");

            // Assert
            result.Should().BeNull();
        }

        [Fact]
        public void Login_ReturnsNull_WhenUserNotFound()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.GetUserByUsername("name")).Returns((User?)null);

            // Act
            var result = _authService.Login("name", "name@test.com", "Password123!");

            // Assert
            result.Should().BeNull();
        }

        [Fact]
        public void Login_ReturnsNull_WhenEmailDoesNotMatch()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Password123!"),
                Role = "User"
            };
            _mockUserRepo.Setup(r => r.GetUserByUsername("name")).Returns(user);

            // Act
            var result = _authService.Login("name", "wrong@test.com", "Password123!");

            // Assert
            result.Should().BeNull();
        }

        [Fact]
        public void Login_ReturnsNull_WhenPasswordIsWrong()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Password123!"),
                Role = "User"
            };
            _mockUserRepo.Setup(r => r.GetUserByUsername("name")).Returns(user);

            // Act
            var result = _authService.Login("name", "name@test.com", "WrongPassword!");

            // Assert
            result.Should().BeNull();
        }

        [Fact]
        public void Login_ReturnsLoginResponse_WhenCredentialsAreValid()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Password123!"),
                Role = "User"
            };
            _mockUserRepo.Setup(r => r.GetUserByUsername("name")).Returns(user);

            // Act
            var result = _authService.Login("name", "name@test.com", "Password123!");

            // Assert
            result.Should().NotBeNull();
            result!.Role.Should().Be("User");
            result.Token.Should().NotBeNullOrEmpty();
            result.Expiration.Should().BeAfter(DateTime.UtcNow);
        }

        [Fact]
        public void Login_ReturnsAdminRole_WhenAdminLogsIn()
        {
            // Arrange
            var adminUser = new User
            {
                Id = 2,
                Username = "admin",
                Email = "admin@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("AdminPass123!"),
                Role = "Admin"
            };
            _mockUserRepo.Setup(r => r.GetUserByUsername("admin")).Returns(adminUser);

            // Act
            var result = _authService.Login("admin", "admin@test.com", "AdminPass123!");

            // Assert
            result.Should().NotBeNull();
            result!.Role.Should().Be("Admin");
        }
        #endregion

        #region ForgotPassword Tests
        [Fact]
        public void ForgotPassword_ReturnsMessageWithoutToken_WhenEmailNotFound()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.GetUserByEmail("unknown@test.com")).Returns((User?)null);

            // Act
            var result = _authService.ForgotPassword("unknown@test.com");

            // Assert
            result.Should().NotBeNull();
            result.ResetToken.Should().BeNull();
            result.Message.Should().NotBeNullOrEmpty();
        }

        [Fact]
        public void ForgotPassword_DoesNotCallUpdateUser_WhenEmailNotFound()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.GetUserByEmail("unknown@test.com")).Returns((User?)null);

            // Act
            _authService.ForgotPassword("unknown@test.com");

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUser(It.IsAny<User>()), Times.Never);
        }

        [Fact]
        public void ForgotPassword_ReturnsTokenAndMessage_WhenEmailExists()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Password123!"),
                Role = "User"
            };
            _mockUserRepo.Setup(r => r.GetUserByEmail("name@test.com")).Returns(user);

            // Act
            var result = _authService.ForgotPassword("name@test.com");

            // Assert
            result.Should().NotBeNull();
            result.ResetToken.Should().NotBeNullOrEmpty();
            result.Message.Should().NotBeNullOrEmpty();
        }

        [Fact]
        public void ForgotPassword_SetsTokenAndExpiry_OnUser_WhenEmailExists()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Password123!"),
                Role = "User"
            };
            _mockUserRepo.Setup(r => r.GetUserByEmail("name@test.com")).Returns(user);

            var before = DateTime.UtcNow;

            // Act
            _authService.ForgotPassword("name@test.com");

            // Assert
            user.PasswordResetToken.Should().NotBeNullOrEmpty();
            user.PasswordResetTokenExpiry.Should().NotBeNull();
            user.PasswordResetTokenExpiry.Should().BeAfter(before);
        }

        [Fact]
        public void ForgotPassword_CallsUpdateUser_WhenEmailExists()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Password123!"),
                Role = "User"
            };
            _mockUserRepo.Setup(r => r.GetUserByEmail("name@test.com")).Returns(user);

            // Act
            _authService.ForgotPassword("name@test.com");

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUser(user), Times.Once);
        }
        #endregion

        #region ResetPassword Tests
        [Fact]
        public void ResetPassword_ReturnsFalse_WhenTokenNotFound()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.GetUserByResetToken("invalid-token")).Returns((User?)null);

            // Act
            var result = _authService.ResetPassword("invalid-token", "NewPassword123!");

            // Assert
            result.Should().BeFalse();
        }

        [Fact]
        public void ResetPassword_ReturnsFalse_WhenTokenIsExpired()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("OldPassword123!"),
                Role = "User",
                PasswordResetToken = "expired-token",
                PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(-1)
            };
            _mockUserRepo.Setup(r => r.GetUserByResetToken("expired-token")).Returns(user);

            // Act
            var result = _authService.ResetPassword("expired-token", "NewPassword123!");

            // Assert
            result.Should().BeFalse();
        }

        [Fact]
        public void ResetPassword_DoesNotCallUpdateUser_WhenTokenIsExpired()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("OldPassword123!"),
                Role = "User",
                PasswordResetToken = "expired-token",
                PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(-1)
            };
            _mockUserRepo.Setup(r => r.GetUserByResetToken("expired-token")).Returns(user);

            // Act
            _authService.ResetPassword("expired-token", "NewPassword123!");

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUser(It.IsAny<User>()), Times.Never);
        }

        [Fact]
        public void ResetPassword_ReturnsTrue_WhenTokenIsValid()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("OldPassword123!"),
                Role = "User",
                PasswordResetToken = "valid-token",
                PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(1)
            };
            _mockUserRepo.Setup(r => r.GetUserByResetToken("valid-token")).Returns(user);

            // Act
            var result = _authService.ResetPassword("valid-token", "NewPassword123!");

            // Assert
            result.Should().BeTrue();
        }

        [Fact]
        public void ResetPassword_HashesNewPassword_WhenTokenIsValid()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("OldPassword123!"),
                Role = "User",
                PasswordResetToken = "valid-token",
                PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(1)
            };
            _mockUserRepo.Setup(r => r.GetUserByResetToken("valid-token")).Returns(user);

            // Act
            _authService.ResetPassword("valid-token", "NewPassword123!");

            // Assert
            user.PasswordHash.Should().NotBe("NewPassword123!");
            user.PasswordHash.Should().StartWith("$2");
            BCrypt.Net.BCrypt.Verify("NewPassword123!", user.PasswordHash).Should().BeTrue();
        }

        [Fact]
        public void ResetPassword_ClearsTokenAndExpiry_WhenTokenIsValid()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("OldPassword123!"),
                Role = "User",
                PasswordResetToken = "valid-token",
                PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(1)
            };
            _mockUserRepo.Setup(r => r.GetUserByResetToken("valid-token")).Returns(user);

            // Act
            _authService.ResetPassword("valid-token", "NewPassword123!");

            // Assert
            user.PasswordResetToken.Should().BeNull();
            user.PasswordResetTokenExpiry.Should().BeNull();
        }

        [Fact]
        public void ResetPassword_CallsUpdateUser_WhenTokenIsValid()
        {
            // Arrange
            var user = new User
            {
                Id = 1,
                Username = "name",
                Email = "name@test.com",
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("OldPassword123!"),
                Role = "User",
                PasswordResetToken = "valid-token",
                PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(1)
            };
            _mockUserRepo.Setup(r => r.GetUserByResetToken("valid-token")).Returns(user);

            // Act
            _authService.ResetPassword("valid-token", "NewPassword123!");

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUser(user), Times.Once);
        }
        #endregion
    }
}