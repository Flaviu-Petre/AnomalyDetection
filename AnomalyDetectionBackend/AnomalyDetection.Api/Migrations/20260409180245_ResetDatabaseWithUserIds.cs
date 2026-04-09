using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace AnomalyDetection.Api.Migrations
{
    /// <inheritdoc />
    public partial class ResetDatabaseWithUserIds : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Username",
                table: "InferenceRecords");

            migrationBuilder.AddColumn<int>(
                name: "UserId",
                table: "InferenceRecords",
                type: "int",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "UserId",
                table: "InferenceRecords");

            migrationBuilder.AddColumn<string>(
                name: "Username",
                table: "InferenceRecords",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }
    }
}
