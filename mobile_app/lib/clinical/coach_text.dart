/// Ukrainian wording for a progression decision
///
/// Built from the action, reason code and evidence numbers, not the English reason.
library;

import 'error_names.dart';

const Map<String, String> _actionsUk = {
  'hold': 'без змін',
  'need_more_sessions': 'потрібно більше занять',
  'safety_hold': 'утримання через ризик для безпеки',
  'reduce_volume': 'зменшення обсягу',
  'restore_volume': 'відновлення обсягу',
  'regress_range': 'зменшення цільової амплітуди',
  'advance_range': 'збільшення цільової амплітуди',
  'unlock_next': 'розблокування наступної вправи',
};

/// The Ukrainian name of a progression action
String actionNameUk(String action) => _actionsUk[action] ?? action;

String _num(Object? v, {int digits = 0}) {
  final d = v is num ? v.toDouble() : double.tryParse('$v');
  return d == null ? '-' : d.toStringAsFixed(digits);
}

/// The Ukrainian explanation of why the dose changed, or did not
///
/// `reason_code` in [evidence] tells apart the outcomes that all report `hold`.
String reasonUk(String action, Map<String, dynamic> evidence) {
  final code = evidence['reason_code'] as String? ?? action;
  final sessions = evidence['n_sessions_considered'] ?? 0;
  final required = evidence['sessions_required'] ?? '-';
  final meanEqi = _num(evidence['mean_eqi_recent']);

  switch (code) {
    case 'need_more_sessions':
      return 'Записано $sessions занять із $required, потрібних для зміни '
          'навантаження.';

    case 'safety_hold':
      final slug = evidence['worst_safety_error'];
      final rate = evidence['worst_safety_rate'];
      final pct = rate is num ? (rate * 100).toStringAsFixed(0) : '-';
      return 'Помилка «${errorNameUk('$slug')}» спрацьовувала у $pct % '
          'повторень останнього заняття.';

    case 'reduce_volume':
      // Negative trend; the wording carries the sign
      final trend = evidence['mean_fatigue_trend'];
      final drop = trend is num ? trend.abs().toStringAsFixed(1) : '-';
      return 'У межах підходу якість знижувалася на $drop бала EQI за '
          'повторення - ознака втоми.';

    case 'regress_range':
      return 'Середній EQI за останні $sessions занять становить $meanEqi - '
          'нижче за поріг зниження складності.';

    case 'at_floor':
      return 'Якість нижча за цільову, але амплітуда вже на нижній межі, '
          'призначеній терапевтом.';

    case 'restore_volume':
      return 'Середній EQI $meanEqi за досі зменшеного обсягу - обсяг можна '
          'повертати до призначеного.';

    case 'advance_range':
      return 'Середній EQI за останні $sessions занять становить $meanEqi - '
          'на рівні порога ускладнення.';

    case 'unlock_next':
      return 'Досягнуто верхньої межі амплітуди за середнього EQI $meanEqi - '
          'можна переходити до наступної вправи.';

    case 'at_ceiling_no_next':
      return 'Амплітуда на верхній межі, а наступної вправи в програмі немає.';

    case 'right_difficulty':
      return 'Середній EQI $meanEqi - складність підібрано правильно.';

    default:
      // A code not handled here
      return 'Підставу рішення не розпізнано ($code).';
  }
}
