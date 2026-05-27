using AnomalyDetection.Api.Repositories;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services.Interfaces;
using System;

namespace AnomalyDetection.Api.Services
{
    public class UserService : IUserService
    {
        #region Fields
        private readonly IUserRepository _userRepo;
        #endregion

        #region Constructor
        public UserService(IUserRepository userRepo)
        {
            _userRepo = userRepo;
        }
        #endregion

        #region Methods
        public object GetAllUsers()
        {
            return _userRepo.GetAllUsers();
        }

        public void UpdateUserRole(string currentUserId, int targetUserId, string newRole)
        {
            if (currentUserId == targetUserId.ToString())
            {
                throw new InvalidOperationException("Security Policy: You cannot change your own role or demote yourself.");
            }

            if (newRole != "Admin" && newRole != "User")
            {
                throw new ArgumentException("Invalid role. Must be 'Admin' or 'User'.");
            }

            _userRepo.UpdateUserRole(targetUserId, newRole);
        }

        public void DeleteUser(string currentUserId, int targetUserId)
        {
            if (currentUserId == targetUserId.ToString())
            {
                throw new InvalidOperationException("Security Policy: You cannot delete your own account.");
            }

            _userRepo.DeleteUser(targetUserId);
        }
        #endregion
    }
}