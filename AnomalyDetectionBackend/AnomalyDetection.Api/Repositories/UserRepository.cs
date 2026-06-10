using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories.Interfaces;

namespace AnomalyDetection.Api.Repositories
{
    public class UserRepository : IUserRepository
    {
        #region Fields
        private readonly AppDbContext _db;
        #endregion

        #region Constructor
        public UserRepository(AppDbContext db)
        {
            _db = db;
        }
        #endregion

        #region Methods 

        #region Get Methods
        public User? GetUserByUsername(string username)
        {
            return _db.Users.FirstOrDefault(u => u.Username == username);
        }
        public User? GetUserByEmail(string email)
        {
            return _db.Users.FirstOrDefault(u => u.Email == email);
        }

        public User? GetUserById(int userId)
        {
            return _db.Users.FirstOrDefault(u => u.Id == userId);
        }

        public List<User> GetAllUsers()
        {
            return _db.Users
                .Select(u => new User { Id = u.Id, Username = u.Username, Role = u.Role })
                .ToList();
        }
        #endregion

        #region Update Methods
        public void AddUser(User user)
        {
            _db.Users.Add(user);
            _db.SaveChanges();
        }

        public void UpdateUserRole(int userId, string newRole)
        {
            var user = _db.Users.FirstOrDefault(u => u.Id == userId);
            if (user != null)
            {
                user.Role = newRole;
                _db.SaveChanges();
            }
        }

        public void UpdateUser(User user)
        {
            _db.Users.Update(user);
            _db.SaveChanges();
        }
        #endregion

        #region Delete Methods
        public void DeleteUser(int userId)
        {
            var user = _db.Users.FirstOrDefault(u => u.Id == userId);
            if (user != null)
            {
                _db.Users.Remove(user);
                _db.SaveChanges();
            }
            else
            {
                throw new KeyNotFoundException("User not found.");
            }
        }
        #endregion

        #region Existence Check
        public bool UserExists(string username)
        {
            return _db.Users.Any(u => u.Username == username);
        }
        #endregion

        #endregion
    }
}
