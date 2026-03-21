using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Services;
using AnomalyDetection.Api.Repositories;
using DotNetEnv;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;

Env.Load();

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

builder.Services.AddSingleton<ModelManagerService>();

builder.Services.AddScoped<UserRepository>(); 

builder.Services.AddScoped<AuthService>();
builder.Services.AddScoped<StatisticsService>();
builder.Services.AddScoped<FeedbackService>();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddOpenApi();

string? connectionString = Environment.GetEnvironmentVariable("DB_CONNECTION");

string jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET") ?? throw new Exception("JWT_SECRET missing in .env");
string jwtIssuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? "AnomalyFactoryApi";
string jwtAudience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? "AnomalyFactoryFrontend";

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = jwtIssuer,
            ValidAudience = jwtAudience,
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret))
        };
    });

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(connectionString));

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseAuthentication();

app.UseAuthorization();

app.MapControllers();

app.Run();