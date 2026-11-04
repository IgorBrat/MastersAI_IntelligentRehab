/// The error taxonomy and the clinician grade scale. `rehab_ml.taxonomy`
library;

enum Severity { none, mild, moderate, severe }

extension SeverityName on Severity {
  /// `Severity.name` on the Python side: upper case
  String get wire => name.toUpperCase();
}

enum ErrorFamily { performance, compensation, control }

/// Error types, in `ERROR_ORDER`. The index is part of the model's output contract
enum ErrorType {
  insufficientRom('insufficient_rom', ErrorFamily.performance,
      'Рух виконано не в повній амплітуді - ви не досягли цільового кута.'),
  incompleteReturn('incomplete_return', ErrorFamily.performance,
      'Ви не повертаєте руку у вихідне положення між повтореннями.'),
  insufficientHold('insufficient_hold', ErrorFamily.performance,
      'Затримка в кінцевій точці коротша за призначену.'),
  trunkLateralLean('trunk_lateral_lean', ErrorFamily.compensation,
      'Ви нахиляєте тулуб убік замість того, щоб піднімати руку.'),
  trunkForwardFlexion('trunk_forward_flexion', ErrorFamily.compensation,
      'Ви нахиляєтеся вперед, щоб дотягнутися, - тримайте спину прямо.'),
  trunkRotation('trunk_rotation', ErrorFamily.compensation,
      'Ви розвертаєте корпус - плечі мають залишатися рівними.'),
  shoulderHiking('shoulder_hiking', ErrorFamily.compensation,
      'Плече піднімається до вуха - опустіть і розслабте плече.',
      safetyRelevant: true),
  elbowFlexionSynergy('elbow_flexion_synergy', ErrorFamily.compensation,
      'Лікоть згинається разом із підйомом руки - тримайте руку прямою.'),
  outOfPlaneDrift('out_of_plane_drift', ErrorFamily.compensation,
      'Рух відходить від заданої площини - ведіть руку чітко у площині.'),
  excessiveSpeed('excessive_speed', ErrorFamily.control,
      'Ви виконуєте рух надто швидко - сповільніться і контролюйте рух.',
      safetyRelevant: true),
  jerkyMovement('jerky_movement', ErrorFamily.control,
      'Рух переривчастий - намагайтеся рухатись плавно, без ривків.');

  const ErrorType(this.slug, this.family, this.patientUk, {this.safetyRelevant = false});

  /// `ErrorSpec.name_short`, the stable wire identifier
  final String slug;
  final ErrorFamily family;

  /// Patient-facing cue, `ErrorSpec.patient`
  final String patientUk;
  final bool safetyRelevant;

  /// `ErrorType.name.lower()` on the Python side, used as the EQI penalty key
  String get pythonName => slug;

  static ErrorType? bySlug(String slug) {
    for (final e in ErrorType.values) {
      if (e.slug == slug) return e;
    }
    return null;
  }
}

/// `taxonomy.ERROR_SLUGS`, the column order of the model's error head
final List<String> errorSlugs = [for (final e in ErrorType.values) e.slug];

const List<double> eqiGradeBoundaries = [25.0, 50.0, 70.0, 88.0];

/// Discretising the quality index into a 0-4 clinician grade
int eqiToGrade(double eqi) {
  var grade = 0;
  for (final bound in eqiGradeBoundaries) {
    if (eqi >= bound) grade++;
  }
  return grade > 4 ? 4 : grade;
}
