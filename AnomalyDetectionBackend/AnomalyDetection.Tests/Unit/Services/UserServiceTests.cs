using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services;
using FluentAssertions;
using Moq;

namespace AnomalyDetection.Tests.Unit.Services
{
    public class UserServiceTests
    {
        #region Setup
        private readonly Mock<IUserRepository> _mockUserRepo;
        private readonly UserService _userService;

        public UserServiceTests()
        {
            _mockUserRepo = new Mock<IUserRepository>();
            _userService = new UserService(_mockUserRepo.Object);
        }
        #endregion

        #region GetAllUsers Tests
        [Fact]
        public void GetAllUsers_ReturnsAllUsers()
        {
            // Arrange
            var users = new List<User>
            {
                new User { Id = 1, Username = "name", Role = "User" },
                new User { Id = 2, Username = "admin", Role = "Admin" }
            };
            _mockUserRepo.Setup(r => r.GetAllUsers()).Returns(users);

            // Act
            var result = _userService.GetAllUsers();

            // Assert
            result.Should().NotBeNull();
            _mockUserRepo.Verify(r => r.GetAllUsers(), Times.Once);
        }

        [Fact]
        public void GetAllUsers_ReturnsEmptyList_WhenNoUsersExist()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.GetAllUsers()).Returns(new List<User>());

            // Act
            var result = _userService.GetAllUsers();

            // Assert
            result.Should().NotBeNull();
            (result as List<User>).Should().BeEmpty();
        }
        #endregion

        #region UpdateUserRole Tests
        [Fact]
        public void UpdateUserRole_CallsRepository_WhenDataIsValid()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.UpdateUserRole(2, "Admin"));

            // Act
            _userService.UpdateUserRole("1", 2, "Admin");

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUserRole(2, "Admin"), Times.Once);
        }

        [Fact]
        public void UpdateUserRole_ThrowsInvalidOperationException_WhenAdminUpdatesOwnRole()
        {
            // Act
            Action act = () => _userService.UpdateUserRole("1", 1, "User");

            // Assert
            act.Should().Throw<InvalidOperationException>()
               .WithMessage("Security Policy: You cannot change your own role or demote yourself.");
        }

        [Fact]
        public void UpdateUserRole_ThrowsArgumentException_WhenRoleIsInvalid()
        {
            // Act
            Action act = () => _userService.UpdateUserRole("1", 2, "SuperAdmin");

            // Assert
            act.Should().Throw<ArgumentException>()
               .WithMessage("Invalid role. Must be 'Admin' or 'User'.");
        }

        [Fact]
        public void UpdateUserRole_DoesNotCallRepository_WhenAdminUpdatesOwnRole()
        {
            // Act
            try { _userService.UpdateUserRole("1", 1, "User"); } catch { }

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUserRole(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        [Fact]
        public void UpdateUserRole_DoesNotCallRepository_WhenRoleIsInvalid()
        {
            // Act
            try { _userService.UpdateUserRole("1", 2, "InvalidRole"); } catch { }

            // Assert
            _mockUserRepo.Verify(r => r.UpdateUserRole(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        [Fact]
        public void UpdateUserRole_AcceptsAdminRole()
        {
            // Act
            Action act = () => _userService.UpdateUserRole("1", 2, "Admin");

            // Assert
            act.Should().NotThrow();
            _mockUserRepo.Verify(r => r.UpdateUserRole(2, "Admin"), Times.Once);
        }

        [Fact]
        public void UpdateUserRole_AcceptsUserRole()
        {
            // Act
            Action act = () => _userService.UpdateUserRole("1", 2, "User");

            // Assert
            act.Should().NotThrow();
            _mockUserRepo.Verify(r => r.UpdateUserRole(2, "User"), Times.Once);
        }
        #endregion

        #region DeleteUser Tests
        [Fact]
        public void DeleteUser_CallsRepository_WhenDataIsValid()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.DeleteUser(2));

            // Act
            _userService.DeleteUser("1", 2);

            // Assert
            _mockUserRepo.Verify(r => r.DeleteUser(2), Times.Once);
        }

        [Fact]
        public void DeleteUser_ThrowsInvalidOperationException_WhenAdminDeletesOwnAccount()
        {
            // Act
            Action act = () => _userService.DeleteUser("1", 1);

            // Assert
            act.Should().Throw<InvalidOperationException>()
               .WithMessage("Security Policy: You cannot delete your own account.");
        }

        [Fact]
        public void DeleteUser_DoesNotCallRepository_WhenAdminDeletesOwnAccount()
        {
            // Act
            try { _userService.DeleteUser("1", 1); } catch { }

            // Assert
            _mockUserRepo.Verify(r => r.DeleteUser(It.IsAny<int>()), Times.Never);
        }

        [Fact]
        public void DeleteUser_PropagatesKeyNotFoundException_WhenUserDoesNotExist()
        {
            // Arrange
            _mockUserRepo.Setup(r => r.DeleteUser(99))
                         .Throws(new KeyNotFoundException("User not found."));

            // Act
            Action act = () => _userService.DeleteUser("1", 99);

            // Assert
            act.Should().Throw<KeyNotFoundException>()
               .WithMessage("User not found.");
        }
        #endregion
    }
}