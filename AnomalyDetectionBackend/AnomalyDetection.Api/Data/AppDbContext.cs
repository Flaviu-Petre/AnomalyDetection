using AnomalyDetection.Api.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace AnomalyDetection.Api.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }
        public DbSet<InferenceRecord> InferenceRecords { get; set; }
        public DbSet<User> Users { get; set; }
    }
}
