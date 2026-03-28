using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace AnomalyDetection.Api.Migrations
{
    /// <inheritdoc />
    public partial class LinkUserToInferences : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Username",
                table: "InferenceRecords",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Username",
                table: "InferenceRecords");
        }
    }
}
