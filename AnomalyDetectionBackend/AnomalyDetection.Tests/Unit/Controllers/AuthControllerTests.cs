using AnomalyDetection.Api.Controllers;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Moq;

namespace AnomalyDetection.Tests.Unit.Controllers
{
    public class AuthControllerTests
    {
        #region Setup
        private readonly Mock<IAuthService> _mockAuthService;
        private readonly Mock<ILogger<AuthController>> _mockLogger;
        private readonly AuthController _controller;

        public AuthControllerTests()
        {
            _mockAuthService = new Mock<IAuthService>();
            _mockLogger = new Mock<ILogger<AuthController>>();
            _controller = new AuthController(_mockAuthService.Object, _mockLogger.Object);
        }
        #endregion

        #region Register Tests
        [Fact]
        public void Register_ReturnsOk_WhenRegistrationIsSuccessful()
        {
            // Arrange
            _mockAuthService.Setup(s => s.IsUsernameTaken("name")).Returns(false);
            _mockAuthService.Setup(s => s.RegisterUser("name", "Password123!", "name@test.com"));

            var request = new RegisterRequest
            {
                Username = "name",
                Password = "Password123!",
                Email = "name@test.com"
            };

            // Act
            var result = _controller.Register(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void Register_ReturnsBadRequest_WhenUsernameIsTaken()
        {
            // Arrange
            _mockAuthService.Setup(s => s.IsUsernameTaken("name")).Returns(true);

            var request = new RegisterRequest
            {
                Username = "name",
                Password = "Password123!",
                Email = "name@test.com"
            };

            // Act
            var result = _controller.Register(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Username already exists.");
        }

        [Fact]
        public void Register_DoesNotCallRegisterUser_WhenUsernameIsTaken()
        {
            // Arrange
            _mockAuthService.Setup(s => s.IsUsernameTaken("name")).Returns(true);

            var request = new RegisterRequest
            {
                Username = "name",
                Password = "Password123!",
                Email = "name@test.com"
            };

            // Act
            _controller.Register(request);

            // Assert
            _mockAuthService.Verify(s => s.RegisterUser(
                It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()
            ), Times.Never);
        }

        [Fact]
        public void Register_ReturnsBadRequest_WhenArgumentExceptionIsThrown()
        {
            // Arrange
            _mockAuthService.Setup(s => s.IsUsernameTaken("name")).Returns(false);
            _mockAuthService.Setup(s => s.RegisterUser(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                            .Throws(new ArgumentException("Invalid email address format."));

            var request = new RegisterRequest
            {
                Username = "name",
                Password = "Password123!",
                Email = "not-an-email"
            };

            // Act
            var result = _controller.Register(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Invalid email address format.");
        }

        [Fact]
        public void Register_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            _mockAuthService.Setup(s => s.IsUsernameTaken("name")).Returns(false);
            _mockAuthService.Setup(s => s.RegisterUser(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                            .Throws(new Exception("Unexpected DB error"));

            var request = new RegisterRequest
            {
                Username = "name",
                Password = "Password123!",
                Email = "name@test.com"
            };

            // Act
            var result = _controller.Register(request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region Login Tests
        [Fact]
        public void Login_ReturnsOk_WhenCredentialsAreValid()
        {
            // Arrange
            var loginResponse = new LoginResponse
            {
                Token = "fake.jwt.token",
                Role = "User",
                Expiration = DateTime.UtcNow.AddHours(8)
            };
            _mockAuthService.Setup(s => s.Login("name", "name@test.com", "Password123!"))
                            .Returns(loginResponse);

            var request = new LoginRequest
            {
                Username = "name",
                Email = "name@test.com",
                Password = "Password123!"
            };

            // Act
            var result = _controller.Login(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(loginResponse);
        }

        [Fact]
        public void Login_ReturnsUnauthorized_WhenCredentialsAreInvalid()
        {
            // Arrange
            _mockAuthService.Setup(s => s.Login(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                            .Returns((LoginResponse?)null);

            var request = new LoginRequest
            {
                Username = "name",
                Email = "name@test.com",
                Password = "WrongPassword!"
            };

            // Act
            var result = _controller.Login(request);

            // Assert
            result.Should().BeOfType<UnauthorizedObjectResult>()
                  .Which.Value.Should().Be("Invalid username or password.");
        }

        [Fact]
        public void Login_ReturnsUnauthorized_WhenServiceReturnsNull()
        {
            // Arrange
            _mockAuthService.Setup(s => s.Login(null, null, It.IsAny<string>()))
                            .Returns((LoginResponse?)null);

            var request = new LoginRequest
            {
                Username = null,
                Email = null,
                Password = "Password123!"
            };

            // Act
            var result = _controller.Login(request);

            // Assert
            result.Should().BeOfType<UnauthorizedObjectResult>();
        }

        [Fact]
        public void Login_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            _mockAuthService.Setup(s => s.Login(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                            .Throws(new Exception("Unexpected error"));

            var request = new LoginRequest
            {
                Username = "name",
                Email = "name@test.com",
                Password = "Password123!"
            };

            // Act
            var result = _controller.Login(request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }

        [Fact]
        public void Login_ReturnsTokenInResponse_WhenLoginSucceeds()
        {
            // Arrange
            var loginResponse = new LoginResponse
            {
                Token = "fake.jwt.token",
                Role = "Admin",
                Expiration = DateTime.UtcNow.AddHours(8)
            };
            _mockAuthService.Setup(s => s.Login(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                            .Returns(loginResponse);

            var request = new LoginRequest
            {
                Username = "name",
                Email = "name@test.com",
                Password = "Password123!"
            };

            // Act
            var result = _controller.Login(request) as OkObjectResult;

            // Assert
            result.Should().NotBeNull();
            var response = result!.Value as LoginResponse;
            response!.Token.Should().Be("fake.jwt.token");
            response.Role.Should().Be("Admin");
        }
        #endregion

        #region ForgotPassword Tests
        [Fact]
        public void ForgotPassword_ReturnsOk_WhenEmailExists()
        {
            // Arrange
            var forgotResponse = new ForgotPasswordResponse
            {
                Message = "Password reset token generated successfully.",
                ResetToken = "abc123token"
            };
            _mockAuthService.Setup(s => s.ForgotPassword("name@test.com"))
                            .Returns(forgotResponse);

            var request = new ForgotPasswordRequest { Email = "name@test.com" };

            // Act
            var result = _controller.ForgotPassword(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(forgotResponse);
        }

        [Fact]
        public void ForgotPassword_ReturnsOk_WhenEmailDoesNotExist()
        {
            // Arrange
            var forgotResponse = new ForgotPasswordResponse
            {
                Message = "If this email is registered, a reset token has been generated.",
                ResetToken = null
            };
            _mockAuthService.Setup(s => s.ForgotPassword("unknown@test.com"))
                            .Returns(forgotResponse);

            var request = new ForgotPasswordRequest { Email = "unknown@test.com" };

            // Act
            var result = _controller.ForgotPassword(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void ForgotPassword_ReturnsBadRequest_WhenEmailIsEmpty()
        {
            // Arrange
            var request = new ForgotPasswordRequest { Email = "" };

            // Act
            var result = _controller.ForgotPassword(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void ForgotPassword_DoesNotCallService_WhenEmailIsEmpty()
        {
            // Arrange
            var request = new ForgotPasswordRequest { Email = "" };

            // Act
            _controller.ForgotPassword(request);

            // Assert
            _mockAuthService.Verify(s => s.ForgotPassword(It.IsAny<string>()), Times.Never);
        }

        [Fact]
        public void ForgotPassword_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            _mockAuthService.Setup(s => s.ForgotPassword(It.IsAny<string>()))
                            .Throws(new Exception("Unexpected DB error"));

            var request = new ForgotPasswordRequest { Email = "name@test.com" };

            // Act
            var result = _controller.ForgotPassword(request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region ResetPassword Tests
        [Fact]
        public void ResetPassword_ReturnsOk_WhenTokenAndPasswordAreValid()
        {
            // Arrange
            _mockAuthService.Setup(s => s.ResetPassword("valid-token", "NewPassword123!"))
                            .Returns(true);

            var request = new ResetPasswordRequest
            {
                Token = "valid-token",
                NewPassword = "NewPassword123!"
            };

            // Act
            var result = _controller.ResetPassword(request);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void ResetPassword_ReturnsBadRequest_WhenTokenIsInvalidOrExpired()
        {
            // Arrange
            _mockAuthService.Setup(s => s.ResetPassword("bad-token", "NewPassword123!"))
                            .Returns(false);

            var request = new ResetPasswordRequest
            {
                Token = "bad-token",
                NewPassword = "NewPassword123!"
            };

            // Act
            var result = _controller.ResetPassword(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Invalid or expired reset token.");
        }

        [Fact]
        public void ResetPassword_ReturnsBadRequest_WhenTokenIsEmpty()
        {
            // Arrange
            var request = new ResetPasswordRequest
            {
                Token = "",
                NewPassword = "NewPassword123!"
            };

            // Act
            var result = _controller.ResetPassword(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void ResetPassword_ReturnsBadRequest_WhenPasswordIsEmpty()
        {
            // Arrange
            var request = new ResetPasswordRequest
            {
                Token = "valid-token",
                NewPassword = ""
            };

            // Act
            var result = _controller.ResetPassword(request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>();
        }

        [Fact]
        public void ResetPassword_DoesNotCallService_WhenTokenIsEmpty()
        {
            // Arrange
            var request = new ResetPasswordRequest
            {
                Token = "",
                NewPassword = "NewPassword123!"
            };

            // Act
            _controller.ResetPassword(request);

            // Assert
            _mockAuthService.Verify(s => s.ResetPassword(It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }

        [Fact]
        public void ResetPassword_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            _mockAuthService.Setup(s => s.ResetPassword(It.IsAny<string>(), It.IsAny<string>()))
                            .Throws(new Exception("Unexpected DB error"));

            var request = new ResetPasswordRequest
            {
                Token = "valid-token",
                NewPassword = "NewPassword123!"
            };

            // Act
            var result = _controller.ResetPassword(request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion
    }
}