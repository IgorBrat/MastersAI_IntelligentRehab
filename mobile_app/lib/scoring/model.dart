/// Hybrid model features (`features.extractor`) and forward pass
/// (`models.clinical_hybrid`) in plain Dart. The EQI head corrects the rule EQI.
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'angles.dart';
import 'exercises.dart';
import 'kinematics.dart';
import 'signal.dart';
import 'taxonomy.dart';

// ---------------------------------------------------------------- features

const int sequenceLength = 64;

/// Linearly resampling every column to exactly [length] frames. `extractor.resample`
List<List<double>> _resampleColumns(List<List<double>> columns, int length) {
  final n = columns.first.length;
  if (n == length) return [for (final c in columns) List<double>.from(c)];
  if (n < 2) return [for (final c in columns) List<double>.filled(length, c.first)];
  final src = linspace(0.0, 1.0, n);
  final dst = linspace(0.0, 1.0, length);
  return [
    for (final c in columns) [for (final q in dst) interp(q, src, c)],
  ];
}

/// `(64, 28)` model input for one repetition, row-major. `extractor.sequence_features`
Float32List sequenceFeatures(AngleSeries angles, {double smoothCutoffHz = 6.0}) {
  var raw = [for (final name in angleChannels) angles[name]];
  if (raw.first.length >= 5) {
    raw = smoothColumns(raw, angles.fps, cutoffHz: smoothCutoffHz);
  }
  final velocity = [for (final c in raw) gradient(c, 1 / angles.fps)];
  final out = _resampleColumns([...raw, ...velocity], sequenceLength);
  // Baseline-centre the positions, not the velocities
  for (var c = 0; c < angleChannels.length; c++) {
    final zero = out[c][0];
    for (var t = 0; t < sequenceLength; t++) {
      out[c][t] -= zero;
    }
  }
  final channels = out.length;
  final flat = Float32List(sequenceLength * channels);
  for (var t = 0; t < sequenceLength; t++) {
    for (var c = 0; c < channels; c++) {
      flat[t * channels + c] = out[c][t];
    }
  }
  return flat;
}

const List<String> summaryFeatureNames = [
  'primary_peak',
  'primary_excursion',
  'primary_rom_ratio',
  'primary_return_error',
  'duration_s',
  'concentric_s',
  'eccentric_s',
  'hold_s',
  'hold_ratio',
  'concentric_eccentric_ratio',
  'sparc',
  'ldlj',
  'n_submovements',
  'peak_angular_velocity',
  'trunk_lean_peak',
  'trunk_flexion_peak',
  'trunk_rotation_peak',
  'shoulder_hike_peak',
  'elbow_coupling_peak',
  'plane_deviation_peak',
  'elbow_flexion_peak',
  'elbow_flexion_mean',
];

/// `(22,)` clinically named vector for one repetition. `extractor.summary_features`
Float32List summaryFeatures(AngleSeries angles, ExerciseDefinition exercise) {
  final primary = angles[exercise.primaryChannel];
  final rom = romMetrics(primary);
  final tempo = tempoMetrics(primary, angles.fps);
  final smoothed = primary.length >= 5 ? smoothSignal(primary, angles.fps) : primary;
  final speed = [for (final v in gradient(smoothed, 1 / angles.fps)) v.abs()];

  final targetExcursion = (exercise.rom.target - rom.start).abs();
  final romRatio = targetExcursion > 1e-9 ? (rom.peak - rom.start).abs() / targetExcursion : 1.0;
  final holdRatio = exercise.tempo.holdS > 1e-9 ? tempo.holdS / exercise.tempo.holdS : 1.0;

  double peak(List<double> s) => s.isEmpty ? 0.0 : percentile([for (final v in s) v.abs()], 90);
  List<double> fromStart(List<double> s) {
    final zero = median(s.sublist(0, math.min(5, s.length)));
    return [for (final v in s) v - zero];
  }

  final elbow = angles['elbow_flexion'];
  final plane = exercise.targetPlaneDeg;
  final values = <String, double>{
    'primary_peak': rom.peak,
    'primary_excursion': rom.excursion,
    'primary_rom_ratio': romRatio.clamp(0.0, 3.0).toDouble(),
    'primary_return_error': rom.returnError,
    'duration_s': tempo.durationS,
    'concentric_s': tempo.concentricS,
    'eccentric_s': tempo.eccentricS,
    'hold_s': tempo.holdS,
    'hold_ratio': holdRatio.clamp(0.0, 3.0).toDouble(),
    'concentric_eccentric_ratio': tempo.concentricEccentricRatio.clamp(0.0, 10.0).toDouble(),
    'sparc': sparc(speed, angles.fps),
    'ldlj': ldlj(speed, angles.fps),
    'n_submovements': nSubmovements(speed, angles.fps).toDouble(),
    'peak_angular_velocity': speed.isEmpty ? 0.0 : percentile(speed, 95),
    'trunk_lean_peak': peak(fromStart(angles.trunkLeanContralateral())),
    'trunk_flexion_peak': peak(fromStart(angles['trunk_forward_flexion'])),
    'trunk_rotation_peak': peak(fromStart(angles['trunk_rotation'])),
    'shoulder_hike_peak': peak(angles.shoulderHike()),
    'elbow_coupling_peak': peak(fromStart(elbow)),
    'plane_deviation_peak':
        peak(plane == null ? List<double>.filled(angles.nFrames, 0) : angles.planeDeviationFrom(plane)),
    'elbow_flexion_peak': elbow.isEmpty ? 0.0 : maxOf(elbow),
    'elbow_flexion_mean': elbow.isEmpty ? 0.0 : mean(elbow),
  };
  // NaN SPARC / LDLJ fall back to the rule engine's neutral values
  if (!values['sparc']!.isFinite) values['sparc'] = exercise.sparcFloor;
  if (!values['ldlj']!.isFinite) values['ldlj'] = 0.0;
  return Float32List.fromList([for (final name in summaryFeatureNames) values[name]!]);
}

// ---------------------------------------------------------------- network

/// `erf`, to float32 precision. Numerical Recipes `erfcc`, fractional error < 1.2e-7
double _erf(double x) {
  final z = x.abs();
  final t = 1.0 / (1.0 + 0.5 * z);
  final erfc = t *
      math.exp(-z * z -
          1.26551223 +
          t *
              (1.00002368 +
                  t *
                      (0.37409196 +
                          t *
                              (0.09678418 +
                                  t *
                                      (-0.18628806 +
                                          t *
                                              (0.27886807 +
                                                  t *
                                                      (-1.13520398 +
                                                          t *
                                                              (1.48851587 +
                                                                  t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1.0 - erfc : erfc - 1.0;
}

/// Exact GELU, `torch.nn.GELU()` with `approximate="none"`
double _gelu(double x) => 0.5 * x * (1.0 + _erf(x / math.sqrt2));

double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

double _tanh(double x) {
  final e = math.exp(2 * x);
  return e.isInfinite ? 1.0 : (e - 1) / (e + 1);
}

/// The hybrid network's weights, BatchNorm already folded into each convolution
class HybridWeights {
  HybridWeights._(this._tensors, this.manifest);

  final Map<String, Float32List> _tensors;

  /// `assets/model/hybrid.json`
  final Map<String, dynamic> manifest;

  /// Reading the float32 blob written by `ml/scripts/export_dart.py`
  factory HybridWeights.fromBytes(ByteData bytes, String manifestJson) {
    final manifest = jsonDecode(manifestJson) as Map<String, dynamic>;
    if (manifest['format'] != 'rehab-ml/dart-weights@1') {
      throw const FormatException('not a rehab-ml Dart weight manifest');
    }
    final total = (manifest['n_floats'] as num).toInt();
    if (bytes.lengthInBytes != total * 4) {
      throw FormatException('weight blob is ${bytes.lengthInBytes} bytes, manifest says ${total * 4}');
    }
    final tensors = <String, Float32List>{};
    for (final entry in (manifest['tensors'] as List).cast<Map<String, dynamic>>()) {
      final shape = (entry['shape'] as List).cast<num>();
      final size = shape.fold<int>(1, (a, b) => a * b.toInt());
      final offset = (entry['offset'] as num).toInt();
      final values = Float32List(size);
      for (var i = 0; i < size; i++) {
        values[i] = bytes.getFloat32((offset + i) * 4, Endian.little);
      }
      tensors[entry['name'] as String] = values;
    }
    // Taxonomy can change without any tensor shape changing
    final slugs = ((manifest['artifact'] as Map?)?['error_slugs'] as List?)?.cast<String>();
    if (slugs != null && slugs.join(',') != errorSlugs.join(',')) {
      throw const FormatException('model was exported against a different error taxonomy');
    }
    return HybridWeights._(tensors, manifest);
  }

  Float32List operator [](String name) {
    final t = _tensors[name];
    if (t == null) throw ArgumentError('weight tensor $name missing');
    return t;
  }

  int get hidden => (manifest['hidden'] as num).toInt();
  int get clinicalHidden => (manifest['clinical_hidden'] as num).toInt();
  int get kernelSize => (manifest['kernel_size'] as num).toInt();
  List<int> get dilations => [for (final d in manifest['dilations'] as List) (d as num).toInt()];
  double get maxEqiCorrection => (manifest['max_eqi_correction'] as num).toDouble();
  double get layerNormEps => (manifest['layer_norm_eps'] as num).toDouble();
  Map<String, dynamic> get artifact => (manifest['artifact'] as Map).cast<String, dynamic>();
}

/// The three outputs `training.export.DeployableModel` returns
class ModelOutput {
  const ModelOutput({required this.correctProb, required this.errorProb, required this.eqi});
  final double correctProb;
  final List<double> errorProb;
  final double eqi;
}

/// Linear layer: `(out, in)` weight times [x]
List<double> _linear(Float32List w, Float32List b, List<double> x) {
  final out = b.length;
  final n = x.length;
  return [
    for (var o = 0; o < out; o++)
      () {
        var s = b[o].toDouble();
        final row = o * n;
        for (var i = 0; i < n; i++) {
          s += w[row + i] * x[i];
        }
        return s;
      }(),
  ];
}

/// Dilated "same" convolution over a time-major `(T, C)` buffer
List<List<double>> _conv(List<List<double>> x, Float32List w, Float32List b, int kernel, int dilation) {
  final steps = x.length;
  final cin = x.first.length;
  final cout = b.length;
  final pad = (kernel - 1) * dilation ~/ 2;
  return [
    for (var t = 0; t < steps; t++)
      () {
        final row = List<double>.filled(cout, 0);
        for (var o = 0; o < cout; o++) {
          var s = b[o].toDouble();
          final base = o * cin * kernel;
          for (var k = 0; k < kernel; k++) {
            final src = t + k * dilation - pad;
            if (src < 0 || src >= steps) continue;
            final input = x[src];
            for (var c = 0; c < cin; c++) {
              s += w[base + c * kernel + k] * input[c];
            }
          }
          row[o] = s;
        }
        return row;
      }(),
  ];
}

/// `models.clinical_hybrid.ClinicalHybridModel.forward`, eval mode
ModelOutput hybridForward(
  HybridWeights w,
  Float32List sequence,
  Float32List summary,
  double baselineEqi,
) {
  final channels = sequence.length ~/ sequenceLength;
  final scale = w['input_norm.scale'];
  final shift = w['input_norm.shift'];
  final projW = w['input_proj.weight'];
  final projB = w['input_proj.bias'];

  // Per-channel standardisation, then the 1x1 input projection
  var h = [
    for (var t = 0; t < sequenceLength; t++)
      _linear(projW, projB, [
        for (var c = 0; c < channels; c++)
          sequence[t * channels + c] * scale[c] + shift[c],
      ]),
  ];

  final kernel = w.kernelSize;
  final dilations = w.dilations;
  for (var i = 0; i < dilations.length; i++) {
    final residual = h;
    var y = _conv(h, w['blocks.$i.conv1.weight'], w['blocks.$i.conv1.bias'], kernel, dilations[i]);
    y = [for (final row in y) [for (final v in row) _gelu(v)]];
    y = _conv(y, w['blocks.$i.conv2.weight'], w['blocks.$i.conv2.bias'], kernel, dilations[i]);
    h = [
      for (var t = 0; t < y.length; t++)
        [for (var c = 0; c < y[t].length; c++) _gelu(y[t][c]) + residual[t][c]],
    ];
  }

  // Mean then max pooling over time
  final hidden = h.first.length;
  final embedding = <double>[
    for (var c = 0; c < hidden; c++) mean([for (final row in h) row[c]]),
    for (var c = 0; c < hidden; c++) maxOf([for (final row in h) row[c]]),
  ];

  // Clinical branch: LayerNorm without affine, then two GELU layers
  final s = [for (final v in summary) v.toDouble()];
  final mu = mean(s);
  final variance = mean([for (final v in s) (v - mu) * (v - mu)]);
  final normed = [for (final v in s) (v - mu) / math.sqrt(variance + w.layerNormEps)];
  var clinical = _linear(w['clinical.0.weight'], w['clinical.0.bias'], normed);
  clinical = [for (final v in clinical) _gelu(v)];
  clinical = _linear(w['clinical.1.weight'], w['clinical.1.bias'], clinical);
  clinical = [for (final v in clinical) _gelu(v)];

  // Gated fusion
  final fused = [...embedding, ...clinical];
  final gate = _linear(w['gate.weight'], w['gate.bias'], fused);
  final gated = [for (var i = 0; i < fused.length; i++) fused[i] * _sigmoid(gate[i])];

  final correct = _linear(w['heads.correct.weight'], w['heads.correct.bias'], gated)[0];
  final errors = _linear(w['heads.errors.weight'], w['heads.errors.bias'], gated);
  final residual = _linear(w['eqi_residual.weight'], w['eqi_residual.bias'], gated)[0];
  final eqi = (baselineEqi + w.maxEqiCorrection * _tanh(residual)).clamp(0.0, 100.0).toDouble();

  return ModelOutput(
    correctProb: _sigmoid(correct),
    errorProb: [for (final e in errors) _sigmoid(e)],
    eqi: eqi,
  );
}

/// One loaded artifact and its metadata. `api.inference.ModelRunner`
class ModelRunner {
  ModelRunner(this.weights);

  final HybridWeights weights;

  String? get exerciseId => weights.artifact['exercise_id'] as String?;
  bool get selfPaced => weights.artifact['self_paced'] as bool? ?? false;

  /// Exercises the artifact was fitted on; a pooled artifact names several
  List<String> get fittedExercises =>
      [for (final e in ((weights.artifact['dataset'] as Map?)?['exercise_ids'] as List? ?? [])) '$e'];

  /// Whether this artifact may score [id]; only exercises it was fitted on
  bool appliesTo(String id) =>
      exerciseId == null ? fittedExercises.isEmpty || fittedExercises.contains(id) : exerciseId == id;

  Map<String, dynamic> info() {
    final a = weights.artifact;
    final reference = (a['loso_reference'] as Map?)?.cast<String, dynamic>() ?? const {};
    return {
      'path': a['path'],
      'architecture': weights.manifest['architecture'],
      'exercise_id': exerciseId,
      'self_paced': selfPaced,
      'corpus': a['corpus'],
      'created_utc': a['created_utc'],
      'fitted_on_subjects': (a['dataset'] as Map?)?['n_subjects'],
      'fitted_on_repetitions': (a['dataset'] as Map?)?['n_samples'],
      'fitted_on_exercises': fittedExercises,
      'mirror_augment': a['mirror_augment'],
      'loso_balanced_accuracy_patients': reference['balanced_accuracy_patients'],
      'loso_report': reference['path'],
      'accuracy_note': a['accuracy'],
      'validated_on_phone': false,
      'runtime': 'dart',
    };
  }

  /// Scoring one repetition window whose rule EQI is already known
  Map<String, dynamic> predict(AngleSeries window, ExerciseDefinition exercise, {required double ruleEqi}) {
    final out = hybridForward(
      weights,
      sequenceFeatures(window),
      summaryFeatures(window, exercise),
      f32(ruleEqi),
    );
    return {
      'correct_prob': out.correctProb,
      'eqi': out.eqi,
      'eqi_rule': ruleEqi,
      'eqi_correction': out.eqi - ruleEqi,
      'errors': {
        for (var i = 0; i < errorSlugs.length; i++) errorSlugs[i]: out.errorProb[i],
      },
      'note': 'Model output. The EQI here is the rule engine\'s EQI plus a bounded '
          'learned correction, never a score from scratch. Not validated on '
          'phone-captured pose.',
    };
  }
}
