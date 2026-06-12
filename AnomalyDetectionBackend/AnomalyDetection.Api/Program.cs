using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.Configuration;
using AnomalyDetection.Api.Repositories;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services;
using AnomalyDetection.Api.Services.Interfaces;
using DotNetEnv;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi;
using Serilog;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

if (builder.Environment.IsDevelopment())
    Env.Load();

builder.Host.UseSerilog((context, configuration) =>
{
    configuration
        .MinimumLevel.Information()
        .MinimumLevel.Override("Microsoft.EntityFrameworkCore.Database.Command", Serilog.Events.LogEventLevel.Warning)
        .WriteTo.Console()
        .WriteTo.File("Logs/anomaly-server-log-.txt", rollingInterval: RollingInterval.Day);
});

builder.Services.AddControllers();

builder.Services.AddSingleton<IModelManagerService, ModelManagerService>();
builder.Services.AddSingleton<IRouterService, RouterService>();

builder.Services.AddScoped<IUserRepository, UserRepository>();
builder.Services.AddScoped<IStatisticsRepository, StatisticsRepository>();

builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<IStatisticsService, StatisticsService>();
builder.Services.AddScoped<IFeedbackService, FeedbackService>();
builder.Services.AddScoped<IUserService, UserService>();
builder.Services.AddScoped<IInferenceService, InferenceService>();

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowNextJsFrontend", policy =>
    {
        policy.WithOrigins("http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

builder.Services.AddEndpointsApiExplorer();

builder.Services.AddSwaggerGen(c =>
{
    c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Type = SecuritySchemeType.Http,
        Scheme = "bearer",
        BearerFormat = "JWT",
        Description = "JWT Authorization header using the Bearer scheme. \r\n\r\nExample: \"eyJhbGci...\""
    });

    c.AddSecurityRequirement(document => new OpenApiSecurityRequirement
    {
        { new OpenApiSecuritySchemeReference("Bearer", document), new List<string>() }
    });
});

builder.Services.AddOpenApi();

string? connectionString = Environment.GetEnvironmentVariable("DB_CONNECTION");

var jwtSettings = new JwtSettings
{
    Secret = Environment.GetEnvironmentVariable("JWT_SECRET") ?? "test-secret-key-fallback-minimum-32chars!!",
    Issuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? "AnomalyFactoryApi",
    Audience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? "AnomalyFactoryFrontend",
    ExpirationHours = int.TryParse(Environment.GetEnvironmentVariable("JWT_EXPIRATION_HOURS"), out var hours) ? hours : 8
};

builder.Services.AddSingleton(Options.Create(jwtSettings));

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = jwtSettings.Issuer,
            ValidAudience = jwtSettings.Audience,
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSettings.Secret))
        };
    });

if (!builder.Environment.IsEnvironment("Testing"))
{
    builder.Services.AddDbContext<AppDbContext>(options =>
        options.UseSqlServer(connectionString));
}

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();
app.MapOpenApi();

// app.UseHttpsRedirection();

app.UseCors("AllowNextJsFrontend");

app.UseAuthentication();

app.UseAuthorization();

app.MapControllers();

if (!app.Environment.IsEnvironment("Testing"))
{
    using var scope = app.Services.CreateScope();
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

    var retries = 10;
    while (retries > 0)
    {
        try
        {
            db.Database.Migrate();
            break;
        }
        catch (Exception ex) when (ex.Message.Contains("already exists"))
        {
            break;
        }
        catch (Exception ex)
        {
            retries--;
            if (retries == 0) throw;
            Console.WriteLine($"DB not ready, retrying... ({retries} attempts left).");
            Thread.Sleep(3000);
        }
    }
}

app.Run();
public partial class Program { }