
const std = @import("std");

pub fn build(b: *std.Build) !void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});
    const no_bin = b.option(bool, "no-bin", "") orelse false;

    const sphtud_dep = b.dependency("sphtud", .{
        .with_gl = true,
        .with_glfw = true,
    });
    const sphtud_mod = sphtud_dep.module("sphtud");
    const exe =b.addExecutable(.{
        .name = "bc",
        .root_source_file = b.path("src/main.zig"),
        .optimize = optimize,
        .target = target,
    });
    exe.addCSourceFile(.{
        .file = b.path("stb_image.c"),
    });
    exe.addIncludePath(b.path("."));
    exe.root_module.addImport("sphtud", sphtud_mod);
    exe.linkLibC();

    if (no_bin) {
        b.getInstallStep().dependOn(&exe.step);
    } else {
        b.installArtifact(exe);
    }

}
