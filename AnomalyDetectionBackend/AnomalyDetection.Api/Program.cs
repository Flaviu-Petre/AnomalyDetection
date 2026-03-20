using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Services;
using DotNetEnv;
using Microsoft.EntityFrameworkCore;

Env.Load();

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

builder.Services.AddSingleton<ModelManagerService>();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddOpenApi();

string? connectionString = Environment.GetEnvironmentVariable("DB_CONNECTION");

builder.Services.AddScoped<StatisticsService>();
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(connectionString));

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
    app.MapOpenApi();
}
app.MapControllers();

app.UseHttpsRedirection();

app.UseAuthorization();

app.Run();