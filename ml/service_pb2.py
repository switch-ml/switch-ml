# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: priv/service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12priv/service.proto\x12\x08switchml\"%\n\x12SendWeightsRequest\x12\x0f\n\x07weights\x18\x01 \x01(\t\"\x15\n\x13SendWeightsResponse\"\x15\n\x13\x46\x65tchWeightsRequest\"\'\n\x14\x46\x65tchWeightsResponse\x12\x0f\n\x07weights\x18\x01 \x01(\t2\xb7\x01\n\x16SwitchmlWeightsService\x12L\n\x0bSendWeights\x12\x1c.switchml.SendWeightsRequest\x1a\x1d.switchml.SendWeightsResponse\"\x00\x12O\n\x0c\x46\x65tchWeights\x12\x1d.switchml.FetchWeightsRequest\x1a\x1e.switchml.FetchWeightsResponse\"\x00\x62\x06proto3')



_SENDWEIGHTSREQUEST = DESCRIPTOR.message_types_by_name['SendWeightsRequest']
_SENDWEIGHTSRESPONSE = DESCRIPTOR.message_types_by_name['SendWeightsResponse']
_FETCHWEIGHTSREQUEST = DESCRIPTOR.message_types_by_name['FetchWeightsRequest']
_FETCHWEIGHTSRESPONSE = DESCRIPTOR.message_types_by_name['FetchWeightsResponse']
SendWeightsRequest = _reflection.GeneratedProtocolMessageType('SendWeightsRequest', (_message.Message,), {
  'DESCRIPTOR' : _SENDWEIGHTSREQUEST,
  '__module__' : 'priv.service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.SendWeightsRequest)
  })
_sym_db.RegisterMessage(SendWeightsRequest)

SendWeightsResponse = _reflection.GeneratedProtocolMessageType('SendWeightsResponse', (_message.Message,), {
  'DESCRIPTOR' : _SENDWEIGHTSRESPONSE,
  '__module__' : 'priv.service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.SendWeightsResponse)
  })
_sym_db.RegisterMessage(SendWeightsResponse)

FetchWeightsRequest = _reflection.GeneratedProtocolMessageType('FetchWeightsRequest', (_message.Message,), {
  'DESCRIPTOR' : _FETCHWEIGHTSREQUEST,
  '__module__' : 'priv.service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FetchWeightsRequest)
  })
_sym_db.RegisterMessage(FetchWeightsRequest)

FetchWeightsResponse = _reflection.GeneratedProtocolMessageType('FetchWeightsResponse', (_message.Message,), {
  'DESCRIPTOR' : _FETCHWEIGHTSRESPONSE,
  '__module__' : 'priv.service_pb2'
  # @@protoc_insertion_point(class_scope:switchml.FetchWeightsResponse)
  })
_sym_db.RegisterMessage(FetchWeightsResponse)

_SWITCHMLWEIGHTSSERVICE = DESCRIPTOR.services_by_name['SwitchmlWeightsService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SENDWEIGHTSREQUEST._serialized_start=32
  _SENDWEIGHTSREQUEST._serialized_end=69
  _SENDWEIGHTSRESPONSE._serialized_start=71
  _SENDWEIGHTSRESPONSE._serialized_end=92
  _FETCHWEIGHTSREQUEST._serialized_start=94
  _FETCHWEIGHTSREQUEST._serialized_end=115
  _FETCHWEIGHTSRESPONSE._serialized_start=117
  _FETCHWEIGHTSRESPONSE._serialized_end=156
  _SWITCHMLWEIGHTSSERVICE._serialized_start=159
  _SWITCHMLWEIGHTSSERVICE._serialized_end=342
# @@protoc_insertion_point(module_scope)
