using System.Security.Claims;

namespace AnomalyDetection.Api.Extensions
{
    public static class ClaimsPrincipalExtensions
    {

        public static int GetUserId(this ClaimsPrincipal user)
        {
            var idStr = user.FindFirstValue("id");
            return int.TryParse(idStr, out var id) ? id : 0;
        }

        public static string? GetUserIdString(this ClaimsPrincipal user)
        {
            return user.FindFirstValue("id");
        }

        public static string GetRole(this ClaimsPrincipal user)
        {
            return user.FindFirstValue(ClaimTypes.Role) ?? "User";
        }
    }
}
