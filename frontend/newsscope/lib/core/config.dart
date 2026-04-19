/// App-wide configuration constants.
///
/// BASE_URL is injected at build time via --dart-define. If the variable
/// is not provided, the production Render URL is used as the default.
class AppConfig {
  AppConfig._();

  static const String baseUrl = String.fromEnvironment(
    'BASE_URL',
    defaultValue: 'https://newsscope-backend.onrender.com',
  );
}
