using AnomalyDetection.Api.Controllers;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Moq;
using System.Security.Claims;

namespace AnomalyDetection.Tests.Unit.Controllers
{
    public class UsersControllerTests
    {
        #region Setup
        private readonly Mock<IUserService> _mockUserService;
        private readonly Mock<ILogger<UsersController>> _mockLogger;
        private readonly UsersController _controller;

        public UsersControllerTests()
        {
            _mockUserService = new Mock<IUserService>();
            _mockLogger = new Mock<ILogger<UsersController>>();
            _controller = new UsersController(_mockUserService.Object, _mockLogger.Object);
        }
        private void SetupAuthenticatedUser(string userId, string role = "Admin")
        {
            var claims = new List<Claim>
            {
                new Claim("id", userId),
                new Claim(ClaimTypes.Role, role)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var principal = new ClaimsPrincipal(identity);

            _controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };
        }
        #endregion

        #region GetAllUsers Tests
        [Fact]
        public void GetAllUsers_ReturnsOk_WithUserList()
        {
            // Arrange
            var users = new List<User>
            {
                new User { Id = 1, Username = "name",  Role = "User" },
                new User { Id = 2, Username = "admin", Role = "Admin" }
            };
            _mockUserService.Setup(s => s.GetAllUsers()).Returns(users);

            // Act
            var result = _controller.GetAllUsers();

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(users);
        }

        [Fact]
        public void GetAllUsers_ReturnsOk_WithEmptyList()
        {
            // Arrange
            _mockUserService.Setup(s => s.GetAllUsers()).Returns(new List<User>());

            // Act
            var result = _controller.GetAllUsers();

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void GetAllUsers_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            _mockUserService.Setup(s => s.GetAllUsers())
                            .Throws(new Exception("DB error"));

            // Act
            var result = _controller.GetAllUsers();

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion

        #region UpdateRole Tests
        [Fact]
        public void UpdateRole_ReturnsOk_WhenRoleIsUpdatedSuccessfully()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.UpdateUserRole("1", 2, "Admin"));

            var request = new UpdateRoleRequest { Role = "Admin" };

            // Act
            var result = _controller.UpdateRole(2, request);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void UpdateRole_ReturnsUnauthorized_WhenUserIdClaimIsMissing()
        {
            // Arrange
            _controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = new ClaimsPrincipal() }
            };

            var request = new UpdateRoleRequest { Role = "Admin" };

            // Act
            var result = _controller.UpdateRole(2, request);

            // Assert
            result.Should().BeOfType<UnauthorizedObjectResult>();
        }

        [Fact]
        public void UpdateRole_ReturnsBadRequest_WhenAdminUpdatesOwnRole()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.UpdateUserRole("1", 1, "User"))
                            .Throws(new InvalidOperationException("Security Policy: You cannot change your own role or demote yourself."));

            var request = new UpdateRoleRequest { Role = "User" };

            // Act
            var result = _controller.UpdateRole(1, request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Security Policy: You cannot change your own role or demote yourself.");
        }

        [Fact]
        public void UpdateRole_ReturnsBadRequest_WhenRoleIsInvalid()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.UpdateUserRole("1", 2, "SuperAdmin"))
                            .Throws(new ArgumentException("Invalid role. Must be 'Admin' or 'User'."));

            var request = new UpdateRoleRequest { Role = "SuperAdmin" };

            // Act
            var result = _controller.UpdateRole(2, request);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Invalid role. Must be 'Admin' or 'User'.");
        }

        [Fact]
        public void UpdateRole_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.UpdateUserRole(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<string>()))
                            .Throws(new Exception("Unexpected error"));

            var request = new UpdateRoleRequest { Role = "Admin" };

            // Act
            var result = _controller.UpdateRole(2, request);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }

        [Fact]
        public void UpdateRole_CallsService_WithCorrectParameters()
        {
            // Arrange
            SetupAuthenticatedUser("1");

            var request = new UpdateRoleRequest { Role = "Admin" };

            // Act
            _controller.UpdateRole(2, request);

            // Assert
            _mockUserService.Verify(s => s.UpdateUserRole("1", 2, "Admin"), Times.Once);
        }
        #endregion

        #region DeleteUser Tests
        [Fact]
        public void DeleteUser_ReturnsOk_WhenUserIsDeletedSuccessfully()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.DeleteUser("1", 2));

            // Act
            var result = _controller.DeleteUser(2);

            // Assert
            result.Should().BeOfType<OkObjectResult>();
        }

        [Fact]
        public void DeleteUser_ReturnsUnauthorized_WhenUserIdClaimIsMissing()
        {
            // Arrange
            _controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = new ClaimsPrincipal() }
            };

            // Act
            var result = _controller.DeleteUser(2);

            // Assert
            result.Should().BeOfType<UnauthorizedObjectResult>();
        }

        [Fact]
        public void DeleteUser_ReturnsBadRequest_WhenAdminDeletesOwnAccount()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.DeleteUser("1", 1))
                            .Throws(new InvalidOperationException("Security Policy: You cannot delete your own account."));

            // Act
            var result = _controller.DeleteUser(1);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Security Policy: You cannot delete your own account.");
        }

        [Fact]
        public void DeleteUser_ReturnsNotFound_WhenUserDoesNotExist()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.DeleteUser("1", 99))
                            .Throws(new KeyNotFoundException("User not found."));

            // Act
            var result = _controller.DeleteUser(99);

            // Assert
            result.Should().BeOfType<NotFoundObjectResult>()
                  .Which.Value.Should().Be("User not found.");
        }

        [Fact]
        public void DeleteUser_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1");
            _mockUserService.Setup(s => s.DeleteUser(It.IsAny<string>(), It.IsAny<int>()))
                            .Throws(new Exception("Unexpected error"));

            // Act
            var result = _controller.DeleteUser(2);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }

        [Fact]
        public void DeleteUser_CallsService_WithCorrectParameters()
        {
            // Arrange
            SetupAuthenticatedUser("1");

            // Act
            _controller.DeleteUser(2);

            // Assert
            _mockUserService.Verify(s => s.DeleteUser("1", 2), Times.Once);
        }
        #endregion
    }
}