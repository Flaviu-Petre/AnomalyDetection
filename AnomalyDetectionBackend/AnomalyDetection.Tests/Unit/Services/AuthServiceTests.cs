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
    }
}