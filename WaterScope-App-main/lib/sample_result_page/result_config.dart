import 'dart:ui';

import 'package:json_annotation/json_annotation.dart';

part 'result_config.g.dart';

@JsonSerializable()
class ResultConfig {
  @JsonKey(fromJson: _colorFromJson, toJson: _colorToJson)
  final Color color;
  final int? maxEColi;
  final int? maxColi;
  final String title;
  final String? resultOverride;
  final String resultText;
  final String explanation;

  const ResultConfig(
    this.color,
    this.maxEColi,
    this.maxColi,
    this.title,
    this.resultOverride,
    this.resultText,
    this.explanation,
  );

  factory ResultConfig.fromJson(Map<String, dynamic> json) =>
      _$ResultConfigFromJson(json);

  Map<String, dynamic> toJson() => _$ResultConfigToJson(this);

  static Color _colorFromJson(String colorString) =>
      Color(int.parse('FF' + colorString, radix: 16));

  static String _colorToJson(Color color) =>
      color.value.toRadixString(16).substring(2);
}
