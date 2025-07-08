const std = @import("std");
const stbi = @cImport({
    @cInclude("stb_image.h");
});

const StbiImage = struct {
    inner: Image(u8),

    fn deinit(self: StbiImage) void {
        stbi.stbi_image_free(self.inner.data.ptr);
    }
};

// FIXME: Generalized image kernel

fn loadImage(path: [:0]const u8) !StbiImage {
    var widthi: c_int = undefined;
    var heighti: c_int = undefined;

    const data = stbi.stbi_load(path, &widthi, &heighti, null, 1);
    errdefer stbi.stbi_image_free(data);

    if (data == null) return error.LoadFailed;

    const width: u31 = @intCast(widthi);
    const height: u31 = @intCast(heighti);

    return .{
        .inner = .{
            .data = data[0..width * height],
            .width = width,
        },
    };
}

// FIXME: only multiply + sum
// FIXME: Needs non-square variants
pub fn ImageKernel(comptime T: type, size: comptime_int) type {
    std.debug.assert(size % 2 == 1);

    return struct {
        inner: [size * size]T,

        const Self = @This();

        // Actually kinda like sum
        fn apply(self: Self, image: anytype, x: u31, y: u31) T {
            const image_height = image.calcHeight();
            var total: T = 0;
            for (0..size) |kernel_y| {
                const image_y = kernelToImagePos(y, image_height, @intCast(kernel_y));
                for (0..size) |kernel_x| {
                    const image_x = kernelToImagePos(x, image.width, @intCast(kernel_x));

                    const mul = self.inner[kernel_y * size + kernel_x];
                    const source = image.data[image_y * image.width + image_x];
                    total += mul * source;
                }
            }

            return total;
        }

        fn kernelToImagePos(center: u31, image_max: u31, kernel_pos: u31) u31 {
            const kernel_center = size / 2;
            if (center < kernel_center) return 0;
            if (center >= image_max - kernel_center) return image_max - 1;

            return center + kernel_pos - kernel_center;
        }
    };
}

pub fn StackMat2D(comptime T: type, comptime height_: usize, comptime width_: usize) type {
    return struct {
        data: [width_ * height_]T,
        const width = width_;
        const height = height_;

        const Self = @This();

        fn sum(self: Self) T {
            var ret: T = 0;
            for (self.data) |v| {
                ret += v;
            }
            return ret;
        }
    };
}

pub fn Image(comptime T: type) type {
    return struct {
        data: []T,
        width: u31,

        const Self = @This();

        fn init(alloc: std.mem.Allocator, width: u31, height: u31) !Self {
            return .{
                .data = try alloc.alloc(T, width * height),
                .width = width,
            };
        }

        fn calcHeight(self: Self) u31 {
            return @intCast(self.data.len / self.width);
        }

        fn dupe(self: Image(u8), alloc: std.mem.Allocator) !Image(u8) {
            return .{
                .data = try alloc.dupe(u8, self.data),
                .width = self.width,
            };
        }

        fn applyKernelAtLocSlice(self: Self, comptime KernelElem: type, kernel: []const KernelElem, x: u31, y: u31, kernel_width: u31, kernel_height: u31, out: []KernelElem) void {
            std.debug.assert(kernel_width % 2 == 1);
            std.debug.assert(kernel_height % 2 == 1);

            const image_height = self.calcHeight();
            const image_width = self.width;
            for (0..kernel_height) |kernel_y| {
                const image_y = kernelToImagePos(y, image_height, @intCast(kernel_y), kernel_height);
                for (0..kernel_width) |kernel_x| {
                    const image_x = kernelToImagePos(x, image_width, @intCast(kernel_x), kernel_width);

                    const mul = kernel[kernel_y * kernel_width + kernel_x];
                    const source = self.data[image_y * image_width + image_x];
                    out[kernel_y * kernel_width + kernel_x] = mul * source;
                }
            }
        }

        fn kernelToImagePos(image_pos: u31, image_max: u31, kernel_pos: u31, kernel_max: u31) u31 {
            const kernel_center = kernel_max / 2;
            if (image_pos < kernel_center) return 0;
            if (image_pos >= image_max - kernel_center) return image_max - 1;

            return image_pos + kernel_pos - kernel_center;
        }

        fn applyKernelAtLoc(self: Self, kernel: anytype, x: u31, y: u31) @TypeOf(kernel) {
            const kernel_data_ti = @typeInfo(@TypeOf(kernel.data));
            const Kernel = @TypeOf(kernel);
            const KernelElem = kernel_data_ti.array.child;

            var ret: Kernel = undefined;
            self.applyKernelAtLocSlice(KernelElem, &kernel.data, x, y, Kernel.width, Kernel.height, &ret.data);
            return ret;
        }
    };
}

const BarcodePreprocessor = struct {
    threshold: u16 = 120,
    in: Image(u8),
    out: Image(u1),

    fn init(alloc: std.mem.Allocator, image: Image(u8)) !BarcodePreprocessor {
        return .{
            .in = image,
            .out = try .init(alloc, image.width, image.calcHeight()),
        };
    }

    fn process(self: BarcodePreprocessor) !void {
        for (0..self.in.calcHeight()) |y| {
            for (0..self.in.width) |x| {
                self.processPixelWBlur(@intCast(x), @intCast(y));
            }
        }
    }

    fn processPixelWBlur(self: BarcodePreprocessor, x: u31, y: u31) void {
        if (x < 2 or x > self.in.width - 2) return;
        if (y < 2 or y > self.in.calcHeight() - 2) return;

        const kernel = ImageKernel(u16, 5){
            .inner = @splat(1),
        };

        const total = kernel.apply(self.in, x, y);

        const out_val: u1 = if (total > self.threshold * 25) 1 else 0;
        self.out.data[y * self.in.width + x] = out_val;
    }

    fn processPixel(self: BarcodePreprocessor, x: usize, y: usize) void {
        const in_val = self.in.data[y * self.in.width + x];
        const out_val: u8 = if (in_val > self.threshold) 255 else 0;
        self.out.data[y * self.in.width + x] = out_val;
    }
};

const BarcodeScanner = struct {
    image: Image(u1),
    debug: Image(DebugPurpose),
    tolerance: u31 = 2,

    const upca_width_modules = 95;
    const start_end_width_modules = 3;

    const HalfUpcCode = [6]u8;
    //const ModulePattern = [7]u8;
    const ModulePattern = [4]u8;
    const numberLut = [_]ModulePattern{
        .{3, 2, 1, 1},
        .{2, 2, 2, 1},
        .{2, 1, 2, 2},
        .{1, 4, 1, 1},
        .{1, 1, 3, 2},
        .{1, 2, 3, 1},
        .{1, 1, 1, 4},
        .{1, 3, 1 ,2},
        .{1, 2, 1, 3},
        .{3, 1, 1, 2},
    };

    fn parseLeft(self: BarcodeScanner, y: u31) ?HalfUpcCode {
        var x, const module_width = (self.findStart(y) orelse return null);
        var ret: HalfUpcCode = undefined;
        for (0..6) |i| {
            if (self.readNumberLeft(&x, y, module_width)) |num| {
                //std.debug.print("Found num: {d}\n", .{num});
                ret[i] = num;
            } else {
                //std.debug.print("Made it {d}\n", .{i});
                return null;
            }
        }
        std.debug.print("Found nubmer: {d}\n", .{ret});
        return ret;
        //const leftLUT = [_]ModulePattern{
        //    .{1, 1, 1, 0, 0, 1, 0},
        //    .{1, 1, 0, 0, 1, 1, 0},
        //    .{1, 1, 0, 1, 1, 0, 0},
        //    .{1, 0, 0, 0, 0, 1, 0},
        //    .{1, 0, 1, 1, 1, 0, 0},
        //    .{1, 0, 0, 1, 1, 1, 0},
        //    .{1, 0, 1, 0, 0, 0, 0},
        //    .{1, 0, 0, 0, 1, 0, 0},
        //    .{1, 0, 0, 1, 0, 0, 0},
        //    .{1, 1, 1, 0, 1, 0, 0},
        //};
        //
    }

    fn readNumberLeft(self: BarcodeScanner, x: *u31, y: u31, module_width: i32) ?u8 {
        const a = self.countLight(x, y);
        const b = self.countDark(x, y);
        const c = self.countLight(x, y);
        const d = self.countDark(x, y);


        var best_err: usize = std.math.maxInt(usize);
        var best_match: u8 = undefined;
        for (numberLut, 0..) |value, idx| {
            var this_err: usize = 0;
            this_err += @abs(a - value[0] * module_width);
            this_err += @abs(b - value[1] * module_width);
            this_err += @abs(c - value[2] * module_width);
            this_err += @abs(d - value[3] * module_width);
            if (this_err < best_err) {
                best_match = @intCast(idx);
                best_err = this_err;
            }
        }

        //std.debug.print("err: {d}\n", .{best_err});
        if (best_err < 10) {
            return best_match;
        }

        return null;
    }

    fn findStart(self: BarcodeScanner, y: u31) ?struct{u31, u31} {
        for (0..self.image.width) |x_us| {
            const x: u31 = @intCast(x_us);
            const module_width = self.isStartEndSequence(x, y) orelse continue;
            const expected_end_start = ((upca_width_modules - start_end_width_modules) * module_width) + x;

            const offs_start = @min(expected_end_start -| 5, self.image.width);
            const offs_end = @min(expected_end_start + 5, self.image.width);

            for (offs_start..offs_end) |end_test_x_us| {
                const end_test_x: u31 = @intCast(end_test_x_us);
                if (self.isStartEndSequence(end_test_x, y)) |_| {
                    const out_x = x + module_width * 4;
                    self.debug.data[y * self.debug.width + out_x] = .barcode_start;
                    return .{out_x, module_width};
                }
            }
        }

        return null;
    }

    // return module width if sequence is found
    fn isStartEndSequence(self: BarcodeScanner, x: u31, y: u31) ?u31 {
        var it = x;

        const start = self.countLight(&it, y);
        if (start != 1) return null;

        const a = self.countDark(&it, y);
        if (a == 0) return null;

        const b = self.countLight(&it, y);
        if (b == 0) return null;

        const c = self.countDark(&it, y);
        if (c == 0) return null;

        if (@abs(a - b) < self.tolerance and @abs(a - c) < self.tolerance) {
            self.debug.data[y * self.debug.width + x] = .potential_barcode_start;
            return @intCast(a);
        }

        return null;
    }

    fn countDark(self: BarcodeScanner, x: *u31, y: u31) i32 {
        const start_x = x.*;
        while (x.* < self.image.width) {
            defer x.* += 1;

            if (self.image.data[y * self.image.width + x.*] != 0) {
                break;
            }
        }
        return @intCast(x.* - start_x);
    }

    // FIXME: hella duped with countDark but whatever
    fn countLight(self: BarcodeScanner, x: *u31, y: u31) i32 {
        const start_x = x.*;
        while (x.* < self.image.width) {
            defer x.* += 1;

            if (self.image.data[y * self.image.width + x.*] == 0) {
                break;
            }
        }
        return @intCast(x.* - start_x);
    }
};


const DebugPurpose = enum {
    none,
    barcode_start,
    potential_barcode_start,
};

fn writePpm(image: Image(u1), debug: Image(DebugPurpose)) !void {
    std.debug.assert(image.width == debug.width);
    std.debug.assert(image.data.len == debug.data.len);

    var ppm = try std.fs.cwd().createFile("out.ppm", .{});
    defer ppm.close();

    var ppm_writer = ppm.writer();
    try ppm_writer.print(
        \\P6
        \\{d} {d}
        \\255
        \\
    , .{image.width, image.calcHeight()});

    for (0..image.calcHeight()) |y| {
        for (0..image.width) |x| {
            const to_write = switch (debug.data[y * image.width + x]) {
                .none => blk: {
                    const val: u8 = if (image.data[y * image.width + x] > 0) 255 else 0;
                    break :blk .{val, val, val};
                },
                .potential_barcode_start => .{ 255, 255, 0 },
                .barcode_start => .{ 0, 0, 255},
            };
            try ppm_writer.writeByte(to_write[0]);
            try ppm_writer.writeByte(to_write[1]);
            try ppm_writer.writeByte(to_write[2]);
        }
    }
}

fn writePpmu8(image: Image(u8), path: []const u8) !void {
    var ppm = try std.fs.cwd().createFile(path, .{});
    defer ppm.close();

    var buf_writer = std.io.bufferedWriter(ppm.writer());
    defer buf_writer.flush() catch {};
    var ppm_writer = buf_writer.writer();

    try ppm_writer.print(
        \\P6
        \\{d} {d}
        \\255
        \\
    , .{image.width, image.calcHeight()});

    for (0..image.calcHeight()) |y| {
        for (0..image.width) |x| {
            const val = image.data[y * image.width + x];
            try ppm_writer.writeByte(val);
            try ppm_writer.writeByte(val);
            try ppm_writer.writeByte(val);
        }
    }
}

fn writePpmi16(image: Image(i16)) !void {
    var ppm = try std.fs.cwd().createFile("out2.ppm", .{});
    defer ppm.close();

    var buf_writer = std.io.bufferedWriter(ppm.writer());
    defer buf_writer.flush() catch {};
    var ppm_writer = buf_writer.writer();
    try ppm_writer.print(
        \\P6
        \\{d} {d}
        \\255
        \\
    , .{image.width, image.calcHeight()});

    for (0..image.calcHeight()) |y| {
        for (0..image.width) |x| {
            const val = image.data[y * image.width + x];
            if (val < 0) {
                try ppm_writer.writeByte(@intCast(std.math.clamp(@abs(val) >> 1, 0, 255)));
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
            } else {
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(@intCast(std.math.clamp(val >> 1, 0, 255)));
            }
        }
    }
}

fn writePpmGrad(original: Image(u8), image: Image(Gradient), out_path: []const u8) !void {
    var ppm = try std.fs.cwd().createFile(out_path, .{});
    defer ppm.close();

    var buf_writer = std.io.bufferedWriter(ppm.writer());
    defer buf_writer.flush() catch {};
    var ppm_writer = buf_writer.writer();
    try ppm_writer.print(
        \\P6
        \\{d} {d}
        \\255
        \\
    , .{image.width, image.calcHeight()});

    for (0..image.calcHeight()) |y| {
        for (0..image.width) |x| {
            const val = image.data[y * image.width + x];
            switch (val.confidence) {
                .no, .maybe => {
                    const original_val = original.data[original.width * y + x];
                    try ppm_writer.writeByte(original_val);
                    try ppm_writer.writeByte(original_val);
                    try ppm_writer.writeByte(original_val);
                    continue;
                },
                else => {},
            }
            const mag = val.magnitude;
            const out = std.math.clamp(mag * 255.0 * 6, 0, 255);
            const x_mul = @abs(@cos(val.direction));
            const y_mul = @abs(@sin(val.direction));
            if (out > 50) {
                try ppm_writer.writeByte(@intFromFloat(x_mul * out));
                try ppm_writer.writeByte(@intFromFloat(y_mul * out));
                try ppm_writer.writeByte(0);

            } else {
                    const original_val = original.data[original.width * y + x];
                    try ppm_writer.writeByte(original_val);
                    try ppm_writer.writeByte(original_val);
                    try ppm_writer.writeByte(original_val);
            }
        }
    }
}

const Gradient = struct {
    // radians[0, 2pi],
    direction: f32,
    magnitude: f32,
    confidence: enum { no, maybe, yes } = .yes,
};

const lower_thresh = 0.05;
const higher_thresh = 0.1;

const SnappedGradDir = enum {
    right,
    up_right,
    up,
    up_left,

    fn fromAngle(angle_in: f32) SnappedGradDir {
        var angle = angle_in;
        // map angle to [0,8], round, then % 4 to get direction
        while (angle < 0) {
            angle += std.math.pi * 2;
        }

        const to_round = angle / 2 / std.math.pi * 8;
        std.debug.assert(angle >= 0);

        const rounded: u8 = @intFromFloat(@round(to_round));
        return @enumFromInt(rounded % 4);
    }
};

test "Grid snap" {
    try std.testing.expectEqual(.right, SnappedGradDir.fromAngle(0));
    try std.testing.expectEqual(.right, SnappedGradDir.fromAngle(std.math.pi));

    try std.testing.expectEqual(.up, SnappedGradDir.fromAngle(std.math.pi / 2.0));
    try std.testing.expectEqual(.up, SnappedGradDir.fromAngle(3.0 * std.math.pi / 2.0));

    try std.testing.expectEqual(.up_right, SnappedGradDir.fromAngle(std.math.pi / 4.0));
    try std.testing.expectEqual(.up_right, SnappedGradDir.fromAngle(std.math.pi / 4.0 * 5.0));

    try std.testing.expectEqual(.up_left, SnappedGradDir.fromAngle(std.math.pi / 4.0 * 3.0));
    try std.testing.expectEqual(.up_left, SnappedGradDir.fromAngle(std.math.pi / 4.0 * 7.0));
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const path = (try std.process.argsAlloc(arena.allocator()))[1];

    const image = try loadImage(path);
    defer image.deinit();

    const gradient = try Image(Gradient).init(arena.allocator(), image.inner.width, image.inner.calcHeight());

    const blur_kernel = StackMat2D(u16, 3, 5) {
        .data = .{
            4, 5, 4,
            9, 12, 9,
            12, 15, 12,
            9, 12, 9,
            4, 5, 4,
        },
    };

    std.debug.print("Applying blur\n", .{});
    const blurry = try Image(u8).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
    for (0..blurry.calcHeight()) |y| {
        for (0..blurry.width) |x| {
            const blurred = image.inner.applyKernelAtLoc(blur_kernel, @intCast(x), @intCast(y)).sum() / blur_kernel.sum();
            blurry.data[y * blurry.width + x] = @intCast(blurred);
        }
    }

    try writePpmu8(blurry, "blurry.ppm");

    const horizontal_gradient_kernel = ImageKernel(i32, 3) {
        .inner = .{
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1,
        },
    };

    const vertical_gradient_kernel = ImageKernel(i32, 3) {
        .inner = .{
             1,  2,  1,
            0, 0, 0,
            -1, -2, -1,
        },
    };

    std.debug.print("Calculating gradient\n", .{});
    for (0..gradient.calcHeight()) |y| {
        for (0..gradient.width) |x| {
            const x_gradient = horizontal_gradient_kernel.apply(blurry, @intCast(x), @intCast(y));
            const y_gradient = vertical_gradient_kernel.apply(blurry, @intCast(x), @intCast(y));
            const x_grad_f: f32 = @floatFromInt(x_gradient);
            const y_grad_f: f32 = @floatFromInt(y_gradient);

            var mag: f32 = @floatFromInt(x_gradient * x_gradient + y_gradient * y_gradient);
            mag = std.math.sqrt(mag);
            mag /= 1024.0;

            const angle = std.math.atan2(y_grad_f, x_grad_f);

            gradient.data[y * gradient.width + x] = .{
                .magnitude = mag,
                .direction = angle,
            };
        }
    }

    std.debug.print("suppression\n", .{});
    const suppressed = try Image(Gradient).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
    for (0..suppressed.calcHeight()) |y| {
        for (0..suppressed.width) |x| {
            const in_grad = gradient.data[y * gradient.width + x];
            const in_dir = SnappedGradDir.fromAngle(in_grad.direction);
            var sample_coords: [2]struct {usize, usize } = switch (in_dir) {
                .right => .{
                    .{x -| 1, y},
                    .{x + 1, y},
                },
                .up => .{
                    .{x , y -| 1},
                    .{x, y + 1},
                },
                .up_left => .{
                    .{x -| 1 , y -| 1},
                    .{x + 1, y + 1},
                },
                .up_right => .{
                    .{x + 1 , y -| 1},
                    .{x -| 1, y + 1},
                },
            };

            suppressed.data[y * gradient.width + x] = in_grad;

            for (&sample_coords) |*sample_coord| {
                sample_coord[0] = @min(gradient.width - 1, sample_coord[0]);
                sample_coord[1] = @min(gradient.calcHeight() - 1, sample_coord[1]);

                const compare = gradient.data[sample_coord[1] * gradient.width + sample_coord[0]];
                const compare_dir = SnappedGradDir.fromAngle(compare.direction);
                if (compare_dir == in_dir and compare.magnitude > in_grad.magnitude) {
                    suppressed.data[y * gradient.width + x].magnitude = 0;
                    break;
                }
            }

            if (in_grad.magnitude < lower_thresh) {
                // FIXME: so much indexing
                suppressed.data[y * gradient.width + x].confidence = .no;
            } else if (in_grad.magnitude < higher_thresh) {
                suppressed.data[y * gradient.width + x].confidence = .maybe;
            } else {
                suppressed.data[y * gradient.width + x].confidence = .yes;
            }
        }
    }

    for (0..suppressed.calcHeight()) |y| {
        for (0..suppressed.width) |x| {
            var y_it  = y -| 1;
            while (y_it <= @min(y + 1, suppressed.calcHeight() - 1)) {
                defer y_it += 1;

                var x_it  = x -| 1;
                while (x_it <= @min(x + 1, suppressed.width - 1)) {
                    defer x_it += 1;

                    if (suppressed.data[y_it * suppressed.width + x_it].confidence == .yes) {
                        // FIXME: technically we are progressing more than 1 pixel
                        suppressed.data[y * suppressed.width + x].confidence = .yes;
                    }
                }
            }
        }
    }

    std.debug.print("Writing to file\n", .{});
    try writePpmGrad(image.inner, gradient, "out2.ppm");
    try writePpmGrad(image.inner, suppressed, "out3.ppm");

    if (false) {
        const debug = try Image(DebugPurpose).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
        @memset(debug.data, .none);

        const processor = try BarcodePreprocessor.init(arena.allocator(), image.inner);
        try processor.process();

        const scanner = BarcodeScanner {
            .image = processor.out,
            .debug = debug,
        };

        for (0..image.inner.calcHeight()) |y_us| {
            const y: u31 = @intCast(y_us);
            _ = scanner.findStart(y);
            _ = scanner.parseLeft(y);
        }

        try writePpm(processor.out, debug);
    }
}
