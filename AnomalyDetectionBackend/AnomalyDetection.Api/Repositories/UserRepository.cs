using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.Entities;

namespace AnomalyDetection.Api.Repositories
{
    public class UserRepository
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
        public bool UserExists(string username)
        {
            return _db.Users.Any(u => u.Username == username);
        }

        public User? GetUserByUsername(string username)
        {
            return _db.Users.FirstOrDefault(u => u.Username == username);
        }

        public void AddUser(User user)
        {
            _db.Users.Add(user);
            _db.SaveChanges();
        }
        #endregion
    }
}
