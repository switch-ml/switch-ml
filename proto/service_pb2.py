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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\x12\x08switchml\"k\n\x12SendWeightsRequest\x12!\n\x07\x66it_res\x18\x01 \x01(\x0b\x32\x10.switchml.FitRes\x12#\n\x08\x65val_res\x18\x02 \x01(\x0b\x32\x11.switchml.EvalRes\x12\r\n\x05round\x18\x03 \x01(\t\"\xa9\x01\n\x13SendWeightsResponse\x12(\n\nparameters\x18\x01 \x01(\x0b\x32\x14.switchml.Parameters\x12\x39\n\x06\x63onfig\x18\x03 \x03(\x0b\x32).switchml.SendWeightsResponse.ConfigEntry\x1a-\n\x0b\x43onfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\x15\n\x13\x46\x65tchWeightsRequest\"\xab\x01\n\x14\x46\x65tchWeightsResponse\x12(\n\nparameters\x18\x01 \x01(\x0b\x32\x14.switchml.Parameters\x12:\n\x06\x63onfig\x18\x03 \x03(\x0b\x32*.switchml.FetchWeightsResponse.ConfigEntry\x1a-\n\x0b\x43onfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"2\n\nParameters\x12\x0f\n\x07tensors\x18\x01 \x03(\x0c\x12\x13\n\x0btensor_type\x18\x02 \x01(\t\"\xa8\x01\n\x06\x46itRes\x12(\n\nparameters\x18\x02 \x01(\x0b\x32\x14.switchml.Parameters\x12\x14\n\x0cnum_examples\x18\x03 \x01(\x03\x12.\n\x07metrics\x18\x04 \x03(\x0b\x32\x1d.switchml.FitRes.MetricsEntry\x1a.\n\x0cMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\x8e\x01\n\x07\x45valRes\x12\x0c\n\x04loss\x18\x01 \x01(\x02\x12\x14\n\x0cnum_examples\x18\x02 \x01(\x03\x12/\n\x07metrics\x18\x03 \x03(\x0b\x32\x1e.switchml.EvalRes.MetricsEntry\x1a.\n\x0cMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x32\xb2\x01\n\x0fSwitchmlService\x12N\n\x0bSendWeights\x12\x1c.switchml.SendWeightsRequest\x1a\x1d.switchml.SendWeightsResponse\"\x00\x30\x01\x12O\n\x0c\x46\x65tchWeights\x12\x1d.switchml.FetchWeightsRequest\x1a\x1e.switchml.FetchWeightsResponse\"\x00\x62\x06proto3')



_SENDWEIGHTSREQUEST = DESCRIPTOR.message_types_by_name['SendWeightsRequest']
_SENDWEIGHTSRESPONSE = DESCRIPTOR.message_types_by_name['SendWeightsResponse']
_SENDWEIGHTSRESPONSE_CONFIGENTRY = _SENDWEIGHTSRESPONSE.nested_types_by_name['ConfigEntry']
_FETCHWEIGHTSREQUEST = DESCRIPTOR.message_types_by_name['FetchWeightsRequest']
_FETCHWEIGHTSRESPONSE = DESCRIPTOR.message_types_by_name['FetchWeightsResponse']
_FETCHWEIGHTSRESPONSE_CONFIGENTRY = _FETCHWEIGHTSRESPONSE.nested_types_by_name['ConfigEntry']
_PARAMETERS = DESCRIPTOR.message_types_by_name['Parameters']
_FITRES = DESCRIPTOR.message_types_by_name['FitRes']
_FITRES_METRICSENTRY = _FITRES.nested_types_by_name['MetricsEntry']
_EVALRES = DESCRIPTOR.message_types_by_name['EvalRes']
_EVALRES_METRICSENTRY = _EVALRES.nested_types_by_name['MetricsEntry']
SendWeightsRequest = _reflection.GeneratedProtocolMessageType('SendWeightsRequest', (_message.Message,), {
  'DESCRIPTOR' : _SENDWEIGHTSREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.SendWeightsRequest)
  })
_sym_db.RegisterMessage(SendWeightsRequest)

SendWeightsResponse = _reflection.GeneratedProtocolMessageType('SendWeightsResponse', (_message.Message,), {

  'ConfigEntry' : _reflection.GeneratedProtocolMessageType('ConfigEntry', (_message.Message,), {
    'DESCRIPTOR' : _SENDWEIGHTSRESPONSE_CONFIGENTRY,
    '__module__' : 'service_pb2'
    # @@protoc_insertion_point(class_scope:switchml.SendWeightsResponse.ConfigEntry)
    })
  ,
  'DESCRIPTOR' : _SENDWEIGHTSRESPONSE,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.SendWeightsResponse)
  })
_sym_db.RegisterMessage(SendWeightsResponse)
_sym_db.RegisterMessage(SendWeightsResponse.ConfigEntry)

FetchWeightsRequest = _reflection.GeneratedProtocolMessageType('FetchWeightsRequest', (_message.Message,), {
  'DESCRIPTOR' : _FETCHWEIGHTSREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FetchWeightsRequest)
  })
_sym_db.RegisterMessage(FetchWeightsRequest)

FetchWeightsResponse = _reflection.GeneratedProtocolMessageType('FetchWeightsResponse', (_message.Message,), {

  'ConfigEntry' : _reflection.GeneratedProtocolMessageType('ConfigEntry', (_message.Message,), {
    'DESCRIPTOR' : _FETCHWEIGHTSRESPONSE_CONFIGENTRY,
    '__module__' : 'service_pb2'
    # @@protoc_insertion_point(class_scope:switchml.FetchWeightsResponse.ConfigEntry)
    })
  ,
  'DESCRIPTOR' : _FETCHWEIGHTSRESPONSE,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FetchWeightsResponse)
  })
_sym_db.RegisterMessage(FetchWeightsResponse)
_sym_db.RegisterMessage(FetchWeightsResponse.ConfigEntry)

Parameters = _reflection.GeneratedProtocolMessageType('Parameters', (_message.Message,), {
  'DESCRIPTOR' : _PARAMETERS,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.Parameters)
  })
_sym_db.RegisterMessage(Parameters)

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

EvalRes = _reflection.GeneratedProtocolMessageType('EvalRes', (_message.Message,), {

  'MetricsEntry' : _reflection.GeneratedProtocolMessageType('MetricsEntry', (_message.Message,), {
    'DESCRIPTOR' : _EVALRES_METRICSENTRY,
    '__module__' : 'service_pb2'
    # @@protoc_insertion_point(class_scope:switchml.EvalRes.MetricsEntry)
    })
  ,
  'DESCRIPTOR' : _EVALRES,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.EvalRes)
  })
_sym_db.RegisterMessage(EvalRes)
_sym_db.RegisterMessage(EvalRes.MetricsEntry)

_SWITCHMLSERVICE = DESCRIPTOR.services_by_name['SwitchmlService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SENDWEIGHTSRESPONSE_CONFIGENTRY._options = None
  _SENDWEIGHTSRESPONSE_CONFIGENTRY._serialized_options = b'8\001'
  _FETCHWEIGHTSRESPONSE_CONFIGENTRY._options = None
  _FETCHWEIGHTSRESPONSE_CONFIGENTRY._serialized_options = b'8\001'
  _FITRES_METRICSENTRY._options = None
  _FITRES_METRICSENTRY._serialized_options = b'8\001'
  _EVALRES_METRICSENTRY._options = None
  _EVALRES_METRICSENTRY._serialized_options = b'8\001'
  _SENDWEIGHTSREQUEST._serialized_start=27
  _SENDWEIGHTSREQUEST._serialized_end=134
  _SENDWEIGHTSRESPONSE._serialized_start=137
  _SENDWEIGHTSRESPONSE._serialized_end=306
  _SENDWEIGHTSRESPONSE_CONFIGENTRY._serialized_start=261
  _SENDWEIGHTSRESPONSE_CONFIGENTRY._serialized_end=306
  _FETCHWEIGHTSREQUEST._serialized_start=308
  _FETCHWEIGHTSREQUEST._serialized_end=329
  _FETCHWEIGHTSRESPONSE._serialized_start=332
  _FETCHWEIGHTSRESPONSE._serialized_end=503
  _FETCHWEIGHTSRESPONSE_CONFIGENTRY._serialized_start=261
  _FETCHWEIGHTSRESPONSE_CONFIGENTRY._serialized_end=306
  _PARAMETERS._serialized_start=505
  _PARAMETERS._serialized_end=555
  _FITRES._serialized_start=558
  _FITRES._serialized_end=726
  _FITRES_METRICSENTRY._serialized_start=680
  _FITRES_METRICSENTRY._serialized_end=726
  _EVALRES._serialized_start=729
  _EVALRES._serialized_end=871
  _EVALRES_METRICSENTRY._serialized_start=680
  _EVALRES_METRICSENTRY._serialized_end=726
  _SWITCHMLSERVICE._serialized_start=874
  _SWITCHMLSERVICE._serialized_end=1052
# @@protoc_insertion_point(module_scope)
