# Generates C++ protobuf + gRPC sources at configure time.
#
# mass_generate_proto(
#     PROTO_DIR <abs path to dir containing .proto files>
#     PROTOS    <relative .proto paths>...
#     OUT_DIR   <abs path for generated .pb.cc/.h>
#     OUT_VAR   <variable name to receive generated source list>
# )
#
# Why configure-time generation: lets us add the .pb.cc files to a regular
# add_library() target, which gives proper dependency tracking, IDE
# integration, and parallel builds. add_custom_command-at-build-time would
# work too but loses some of that.

function(mass_generate_proto)
    set(options)
    set(oneValueArgs PROTO_DIR OUT_DIR OUT_VAR)
    set(multiValueArgs PROTOS)
    cmake_parse_arguments(MGP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT MGP_PROTO_DIR OR NOT MGP_OUT_DIR OR NOT MGP_OUT_VAR OR NOT MGP_PROTOS)
        message(FATAL_ERROR "mass_generate_proto: all of PROTO_DIR / OUT_DIR / OUT_VAR / PROTOS are required")
    endif()

    file(MAKE_DIRECTORY "${MGP_OUT_DIR}")

    # Locate the protoc executable + the grpc_cpp plugin. vcpkg's gRPC port
    # exposes these as imported targets; resolving via $<TARGET_FILE:...> is
    # not safe at configure time, so we read their LOCATION property.
    if(NOT TARGET protobuf::protoc)
        message(FATAL_ERROR "protobuf::protoc target missing — install protobuf via vcpkg")
    endif()
    if(NOT TARGET gRPC::grpc_cpp_plugin)
        message(FATAL_ERROR "gRPC::grpc_cpp_plugin target missing — install gRPC via vcpkg")
    endif()

    get_target_property(_protoc_exe protobuf::protoc IMPORTED_LOCATION_RELEASE)
    if(NOT _protoc_exe)
        get_target_property(_protoc_exe protobuf::protoc IMPORTED_LOCATION)
    endif()

    get_target_property(_grpc_plugin gRPC::grpc_cpp_plugin IMPORTED_LOCATION_RELEASE)
    if(NOT _grpc_plugin)
        get_target_property(_grpc_plugin gRPC::grpc_cpp_plugin IMPORTED_LOCATION)
    endif()

    set(_generated_sources "")
    foreach(_proto IN LISTS MGP_PROTOS)
        get_filename_component(_proto_name "${_proto}" NAME_WE)
        get_filename_component(_proto_dir "${_proto}" DIRECTORY)

        if(_proto_dir)
            file(MAKE_DIRECTORY "${MGP_OUT_DIR}/${_proto_dir}")
            set(_pb_h    "${MGP_OUT_DIR}/${_proto_dir}/${_proto_name}.pb.h")
            set(_pb_cc   "${MGP_OUT_DIR}/${_proto_dir}/${_proto_name}.pb.cc")
            set(_grpc_h  "${MGP_OUT_DIR}/${_proto_dir}/${_proto_name}.grpc.pb.h")
            set(_grpc_cc "${MGP_OUT_DIR}/${_proto_dir}/${_proto_name}.grpc.pb.cc")
        else()
            set(_pb_h    "${MGP_OUT_DIR}/${_proto_name}.pb.h")
            set(_pb_cc   "${MGP_OUT_DIR}/${_proto_name}.pb.cc")
            set(_grpc_h  "${MGP_OUT_DIR}/${_proto_name}.grpc.pb.h")
            set(_grpc_cc "${MGP_OUT_DIR}/${_proto_name}.grpc.pb.cc")
        endif()

        add_custom_command(
            OUTPUT  "${_pb_h}" "${_pb_cc}" "${_grpc_h}" "${_grpc_cc}"
            COMMAND "${_protoc_exe}"
                    "-I=${MGP_PROTO_DIR}"
                    "--cpp_out=${MGP_OUT_DIR}"
                    "--grpc_out=${MGP_OUT_DIR}"
                    "--plugin=protoc-gen-grpc=${_grpc_plugin}"
                    "${_proto}"
            DEPENDS "${MGP_PROTO_DIR}/${_proto}"
            WORKING_DIRECTORY "${MGP_PROTO_DIR}"
            COMMENT "Generating C++ proto/gRPC sources for ${_proto}"
            VERBATIM
        )

        list(APPEND _generated_sources "${_pb_cc}" "${_grpc_cc}")
    endforeach()

    set(${MGP_OUT_VAR} "${_generated_sources}" PARENT_SCOPE)
endfunction()
