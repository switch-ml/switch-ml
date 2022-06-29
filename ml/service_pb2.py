# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\x12\x08switchml\"`\n\x12SendWeightsRequest\x12!\n\x07\x66it_res\x18\x02 \x01(\x0b\x32\x10.switchml.FitRes\x12\'\n\x08\x65val_res\x18\x03 \x01(\x0b\x32\x15.switchml.EvaluateRes\"\x15\n\x13SendWeightsResponse\"\x15\n\x13\x46\x65tchWeightsRequest\"@\n\x14\x46\x65tchWeightsResponse\x12(\n\nparameters\x18\x01 \x01(\x0b\x32\x14.switchml.Parameters\"2\n\nParameters\x12\x0f\n\x07tensors\x18\x01 \x03(\x0c\x12\x13\n\x0btensor_type\x18\x02 \x01(\t\"\x96\x01\n\x0b\x45valuateRes\x12\x0c\n\x04loss\x18\x02 \x01(\x02\x12\x14\n\x0cnum_examples\x18\x03 \x01(\x03\x12\x33\n\x07metrics\x18\x04 \x03(\x0b\x32\".switchml.EvaluateRes.MetricsEntry\x1a.\n\x0cMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\xa8\x01\n\x06\x46itRes\x12(\n\nparameters\x18\x02 \x01(\x0b\x32\x14.switchml.Parameters\x12\x14\n\x0cnum_examples\x18\x03 \x01(\x03\x12.\n\x07metrics\x18\x04 \x03(\x0b\x32\x1d.switchml.FitRes.MetricsEntry\x1a.\n\x0cMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x32\xb7\x01\n\x16SwitchmlWeightsService\x12L\n\x0bSendWeights\x12\x1c.switchml.SendWeightsRequest\x1a\x1d.switchml.SendWeightsResponse\"\x00\x12O\n\x0c\x46\x65tchWeights\x12\x1d.switchml.FetchWeightsRequest\x1a\x1e.switchml.FetchWeightsResponse\"\x00\x62\x06proto3')



_SENDWEIGHTSREQUEST = DESCRIPTOR.message_types_by_name['SendWeightsRequest']
_SENDWEIGHTSRESPONSE = DESCRIPTOR.message_types_by_name['SendWeightsResponse']
_FETCHWEIGHTSREQUEST = DESCRIPTOR.message_types_by_name['FetchWeightsRequest']
_FETCHWEIGHTSRESPONSE = DESCRIPTOR.message_types_by_name['FetchWeightsResponse']
_PARAMETERS = DESCRIPTOR.message_types_by_name['Parameters']
_EVALUATERES = DESCRIPTOR.message_types_by_name['EvaluateRes']
_EVALUATERES_METRICSENTRY = _EVALUATERES.nested_types_by_name['MetricsEntry']
_FITRES = DESCRIPTOR.message_types_by_name['FitRes']
_FITRES_METRICSENTRY = _FITRES.nested_types_by_name['MetricsEntry']
SendWeightsRequest = _reflection.GeneratedProtocolMessageType('SendWeightsRequest', (_message.Message,), {
  'DESCRIPTOR' : _SENDWEIGHTSREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.SendWeightsRequest)
  })
_sym_db.RegisterMessage(SendWeightsRequest)

SendWeightsResponse = _reflection.GeneratedProtocolMessageType('SendWeightsResponse', (_message.Message,), {
  'DESCRIPTOR' : _SENDWEIGHTSRESPONSE,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.SendWeightsResponse)
  })
_sym_db.RegisterMessage(SendWeightsResponse)

FetchWeightsRequest = _reflection.GeneratedProtocolMessageType('FetchWeightsRequest', (_message.Message,), {
  'DESCRIPTOR' : _FETCHWEIGHTSREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FetchWeightsRequest)
  })
_sym_db.RegisterMessage(FetchWeightsRequest)

FetchWeightsResponse = _reflection.GeneratedProtocolMessageType('FetchWeightsResponse', (_message.Message,), {
  'DESCRIPTOR' : _FETCHWEIGHTSRESPONSE,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FetchWeightsResponse)
  })
_sym_db.RegisterMessage(FetchWeightsResponse)

Parameters = _reflection.GeneratedProtocolMessageType('Parameters', (_message.Message,), {
  'DESCRIPTOR' : _PARAMETERS,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.Parameters)
  })
_sym_db.RegisterMessage(Parameters)

EvaluateRes = _reflection.GeneratedProtocolMessageType('EvaluateRes', (_message.Message,), {

  'MetricsEntry' : _reflection.GeneratedProtocolMessageType('MetricsEntry', (_message.Message,), {
    'DESCRIPTOR' : _EVALUATERES_METRICSENTRY,
    '__module__' : 'service_pb2'
    # @@protoc_insertion_point(class_scope:switchml.EvaluateRes.MetricsEntry)
    })
  ,
  'DESCRIPTOR' : _EVALUATERES,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.EvaluateRes)
  })
_sym_db.RegisterMessage(EvaluateRes)
_sym_db.RegisterMessage(EvaluateRes.MetricsEntry)

FitRes = _reflection.GeneratedProtocolMessageType('FitRes', (_message.Message,), {

  'MetricsEntry' : _reflection.GeneratedProtocolMessageType('MetricsEntry', (_message.Message,), {
    'DESCRIPTOR' : _FITRES_METRICSENTRY,
    '__module__' : 'service_pb2'
    # @@protoc_insertion_point(class_scope:switchml.FitRes.MetricsEntry)
    })
  ,
  'DESCRIPTOR' : _FITRES,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FitRes)
  })
_sym_db.RegisterMessage(FitRes)
_sym_db.RegisterMessage(FitRes.MetricsEntry)

_SWITCHMLWEIGHTSSERVICE = DESCRIPTOR.services_by_name['SwitchmlWeightsService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EVALUATERES_METRICSENTRY._options = None
  _EVALUATERES_METRICSENTRY._serialized_options = b'8\001'
  _FITRES_METRICSENTRY._options = None
  _FITRES_METRICSENTRY._serialized_options = b'8\001'
  _SENDWEIGHTSREQUEST._serialized_start=27
  _SENDWEIGHTSREQUEST._serialized_end=123
  _SENDWEIGHTSRESPONSE._serialized_start=125
  _SENDWEIGHTSRESPONSE._serialized_end=146
  _FETCHWEIGHTSREQUEST._serialized_start=148
  _FETCHWEIGHTSREQUEST._serialized_end=169
  _FETCHWEIGHTSRESPONSE._serialized_start=171
  _FETCHWEIGHTSRESPONSE._serialized_end=235
  _PARAMETERS._serialized_start=237
  _PARAMETERS._serialized_end=287
  _EVALUATERES._serialized_start=290
  _EVALUATERES._serialized_end=440
  _EVALUATERES_METRICSENTRY._serialized_start=394
  _EVALUATERES_METRICSENTRY._serialized_end=440
  _FITRES._serialized_start=443
  _FITRES._serialized_end=611
  _FITRES_METRICSENTRY._serialized_start=394
  _FITRES_METRICSENTRY._serialized_end=440
  _SWITCHMLWEIGHTSSERVICE._serialized_start=614
  _SWITCHMLWEIGHTSSERVICE._serialized_end=797
# @@protoc_insertion_point(module_scope)
