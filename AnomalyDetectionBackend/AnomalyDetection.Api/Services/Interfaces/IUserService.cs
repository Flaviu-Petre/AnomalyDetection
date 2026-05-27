namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IUserService
    {
        object GetAllUsers();
        void UpdateUserRole(string currentUserId, int targetUserId, string newRole);
        void DeleteUser(string currentUserId, int targetUserId);
    }
}