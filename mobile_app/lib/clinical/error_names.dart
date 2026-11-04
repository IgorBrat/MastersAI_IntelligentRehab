/// Ukrainian names for all eleven error types in `rehab_ml.taxonomy`
///
/// `watched_errors` holds only an exercise's compensations, so this is the fallback.
library;

/// A short label, for lists and headings
const Map<String, String> _namesUk = {
  'insufficient_rom': 'Неповна амплітуда руху',
  'incomplete_return': 'Неповне повернення у вихідне положення',
  'insufficient_hold': 'Закоротка затримка в кінцевій точці',
  'trunk_lateral_lean': 'Боковий нахил тулуба',
  'trunk_forward_flexion': 'Нахил тулуба вперед',
  'trunk_rotation': 'Розворот корпуса',
  'shoulder_hiking': 'Підйом плечового поясу',
  'elbow_flexion_synergy': 'Згинання ліктя під час підйому',
  'out_of_plane_drift': 'Відхилення від площини руху',
  'excessive_speed': 'Надто швидкий рух',
  'jerky_movement': 'Переривчастий рух',
};

/// The corrective sentence for the patient, mirroring `ERROR_SPECS[...].patient`
const Map<String, String> _cuesUk = {
  'insufficient_rom':
      'Рух виконано не в повній амплітуді - ви не досягли цільового кута.',
  'incomplete_return':
      'Ви не повертаєте руку у вихідне положення між повтореннями.',
  'insufficient_hold': 'Затримка в кінцевій точці коротша за призначену.',
  'trunk_lateral_lean':
      'Ви нахиляєте тулуб убік замість того, щоб піднімати руку.',
  'trunk_forward_flexion':
      'Ви нахиляєтеся вперед, щоб дотягнутися, - тримайте спину прямо.',
  'trunk_rotation': 'Ви розвертаєте корпус - плечі мають залишатися рівними.',
  'shoulder_hiking': 'Плече піднімається до вуха - опустіть і розслабте плече.',
  'elbow_flexion_synergy':
      'Лікоть згинається разом із підйомом руки - тримайте руку прямою.',
  'out_of_plane_drift':
      'Рух відходить від заданої площини - ведіть руку чітко у площині.',
  'excessive_speed':
      'Ви виконуєте рух надто швидко - сповільніться і контролюйте рух.',
  'jerky_movement':
      'Рух переривчастий - намагайтеся рухатись плавно, без ривків.',
};

/// The short Ukrainian name of [slug]; an unknown slug is returned unchanged
String errorNameUk(String slug) => _namesUk[slug] ?? slug;

/// The corrective sentence for [slug], preferring the cue in [watched]
String errorCueUk(String slug, {Map<String, String>? watched}) {
  final specific = watched?[slug];
  if (specific != null && specific.isNotEmpty) return specific;
  return _cuesUk[slug] ?? errorNameUk(slug);
}
