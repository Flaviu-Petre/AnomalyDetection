using AnomalyDetection.Api.Models.Entities;

namespace AnomalyDetection.Api.Repositories.Interfaces
{
    public interface IUserRepository
    {
        User? GetUserByUsername(string username);
        User? GetUserByEmail(string email);
        User? GetUserById(int userId);
        List<User> GetAllUsers();
        void AddUser(User user);
        void UpdateUserRole(int userId, string newRole);
        void UpdateUser(User user);
        void DeleteUser(int userId);
        bool UserExists(string username);
    }
}