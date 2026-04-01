using AnomalyDetection.Api.Repositories;
using System;

namespace AnomalyDetection.Api.Services
{
    public class UserService
    {
        private readonly UserRepository _userRepo;

        public UserService(UserRepository userRepo)
        {
            _userRepo = userRepo;
        }

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
    }
}