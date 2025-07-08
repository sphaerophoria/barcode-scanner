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

const BarcodeScanner2 = struct {
    row: []const RleGenerator.RleItem,
    debug: []DebugPurpose,

    const length_variation_allowed: usize = 2;

    // Returns RLE index
    fn findStartSequence(self: BarcodeScanner2) ?usize {
        if (self.row.len < 3) return null;
        for (0..self.row.len - 3) |rle_idx| {
            const start = self.findStartEndSequence(rle_idx) orelse continue;

            // 4 transitions per character, 12 chars, + 5 in the middle sequence + 3 at the start
            // - 1 for error tolerance
            const potential_end_rle_idx = start.rle_idx + 4 * 12 + 5 + 3 - 1;

            if (potential_end_rle_idx > self.row.len) continue;
            for (potential_end_rle_idx..self.row.len) |search_idx| {
                if (self.findStartEndSequence(search_idx)) |end| {
                    // 7 modules per character, 12 chars, 5 modules  middle, 3 start, 3 end
                    // end_rle_idx is start of the end segment, so drop last 3
                    const expected_width_modules = 7 * 12 + 5 + 3;
                    const width_px = countRleSizePx(self.row[start.rle_idx..end.rle_idx]);

                    // Check if within some tolerance
                    const expected_width_px = expected_width_modules * start.module_width;
                    const err_px = diff(expected_width_px, width_px);
                    const err_percent = err_px * 100 / expected_width_px;
                    std.debug.print("err: {d} rle_width: {d}\n" ,.{err_percent, end.rle_idx - start.rle_idx});
                    // FIXME: Check that end module width is ~= start module width
                    // FIXME: Constrain search to that module width
                    if (err_percent < 20) {
                        const x_px: usize = countRleSizePx(self.row[0..start.rle_idx]);
                        self.debug[x_px] = .barcode_start;
                        self.debug[x_px + width_px] = .barcode_end;
                        return rle_idx;
                    }
                }
            }
        }

        return null;
    }

    const StartEndSeq = struct {
        module_width: usize,
        rle_idx: usize,
    };

    fn findStartEndSequence(self: BarcodeScanner2, start_idx: usize) ?StartEndSeq {
        if (self.row.len < 3) return null;
        if (self.row.len - 3 < start_idx) return null;
        var x_px: usize = countRleSizePx(self.row[0..start_idx]);
        for (start_idx..self.row.len - 3) |rle_idx| {
            const a = self.row[rle_idx];

            defer x_px += a.length;
            const b = self.row[rle_idx + 1];
            const c = self.row[rle_idx + 2];

            if (b.val < a.val) continue;
            if (c.val > b.val) continue;

            if (diff(a.length, b.length) > length_variation_allowed) continue;
            if (diff(b.length, c.length) > length_variation_allowed) continue;

            self.debug[x_px] = .potential_barcode_start;
            return .{
                .module_width = a.length,
                .rle_idx = rle_idx,
            };
        }

        return null;
    }

    fn countRleSizePx(row_segment: []const RleGenerator.RleItem) usize {
        var total: usize = 0;
        for (row_segment) |elem| {
            total += elem.length;
        }
        return total;
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
    barcode_end,
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
                .barcode_end => .{ 255, 0, 0},
            };
            try ppm_writer.writeByte(to_write[0]);
            try ppm_writer.writeByte(to_write[1]);
            try ppm_writer.writeByte(to_write[2]);
        }
    }
}

fn writeDebug(image: Image(u8), debug: Image(DebugPurpose)) !void {
    std.debug.assert(image.width == debug.width);
    std.debug.assert(image.data.len == debug.data.len);

    var ppm = try std.fs.cwd().createFile("out.ppm", .{});
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
            const to_write = switch (debug.data[y * image.width + x]) {
                .none => blk: {
                    const val: u8 = image.data[y * image.width + x];
                    break :blk .{val, val, val};
                },
                .potential_barcode_start => .{ 255, 255, 0 },
                .barcode_start => .{ 0, 0, 255},
                .barcode_end => .{ 255, 0, 0},
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

fn writeEdges(original: Image(u8), edges: Image(u32), out_path: []const u8) !void {
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
    , .{edges.width, edges.calcHeight()});

    for (0..edges.calcHeight()) |y| {
        for (0..edges.width) |x| {
            const val = edges.data[y * edges.width + x];
            if (val == 0) {
                const source_val = original.data[y * original.width + x];
                try ppm_writer.writeByte(source_val);
                try ppm_writer.writeByte(source_val);
                try ppm_writer.writeByte(source_val);

            } else {
                try ppm_writer.writeByte(@intCast(@min(val, 255)));
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
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

fn RollingAverage(comptime T: type, comptime size: comptime_int) type {
    return struct {
        data: [size]T = undefined,
        idx: usize = 0,
        used: usize = 0,

        const Self = @This();
        fn push(self: *Self, val: T) void {
            self.data[self.idx] = val;

            if (self.used < size) {
                self.used += 1;
            }
            self.idx = (self.idx + 1) % size;
        }

        fn average(self: Self) T {
            var ret: T = 0;
            for (&self.data) |v| {
                ret += v;
            }

            return @intCast(ret / self.used);
        }
    };
}

fn diff(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (a > b) return a - b else return b - a;
}

const RleGenerator = struct {
    row: std.ArrayList(RleItem),
    min: u8 = std.math.maxInt(u8),
    max: u8 = 0,
    num_entries: usize = 0,

    const EdgeDir = enum {
        light_to_dark,
        dark_to_light,
    };

    const RleItem = struct {
        val: u8,
        length: usize,
    };

    fn init(alloc: std.mem.Allocator) RleGenerator {
        return .{
            .row = .init(alloc),
        };
    }

    fn pushPixel(self: *RleGenerator, val: u8) void {
        self.min = @min(val, self.min);
        self.max = @max(val, self.max);
        self.num_entries += 1;
    }

    fn markEdge(self: *RleGenerator, edge_dir: EdgeDir) !void {
        if (self.num_entries == 0) return;

        const to_push = switch (edge_dir) {
            .light_to_dark => self.max,
            .dark_to_light => self.min,

        };
        try self.row.append(.{
            .val = to_push,
            .length = self.num_entries,
        });

        self.min = std.math.maxInt(u8);
        self.max = 0;
        self.num_entries = 0;
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const path = (try std.process.argsAlloc(arena.allocator()))[1];

    const image = try loadImage(path);
    defer image.deinit();

    //const gradient = try Image(Gradient).init(arena.allocator(), image.inner.width, image.inner.calcHeight());

    const blur_kernel = StackMat2D(u16, 1, 5) {
        .data = .{
            5,
            12,
            15,
            12,
            5,
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

    //try writePpmu8(blurry, "blurry.ppm");

    const horizontal_gradient_kernel = StackMat2D(i32, 3, 3) {
        .data = .{
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1,
        },
    };

    const edges = try Image(u32).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
    @memset(edges.data, 0);

    const rle_back_and_forth = try Image(u8).init(arena.allocator(), image.inner.width, image.inner.calcHeight());

    const debug = try Image(DebugPurpose).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
    @memset(debug.data, .none);

    var gradient_average = RollingAverage(u32, 1){};
    const gradient_thresh = 5;
    std.debug.print("Calculating gradient\n", .{});
    for (0..edges.calcHeight()) |y| {
        var row_rle = RleGenerator.init(arena.allocator());
        var last_average: u32 = 0;
        var increasing: bool = true;

        for (0..edges.width) |x| {
            row_rle.pushPixel(image.inner.data[y * image.inner.width + x]);
            const this_gradient = blurry.applyKernelAtLoc(horizontal_gradient_kernel, @intCast(x), @intCast(y)).sum();
            const gradient_dir: RleGenerator.EdgeDir = if (this_gradient < 0) .light_to_dark else .dark_to_light;

            gradient_average.push(@abs(this_gradient));
            const averaged = gradient_average.average();
            if (averaged < last_average and increasing and last_average > gradient_thresh) {
                try row_rle.markEdge(gradient_dir);
                edges.data[y * edges.width + x - 1] = last_average;
            }

            if (averaged <= last_average) {
                increasing = false;
            } else {
                increasing = true;
            }

            last_average = averaged;
        }

        // FIXME: Abusing info that outside should always be light
        try row_rle.markEdge(.light_to_dark);

        var x: usize = 0;
        var length_sum: usize = 0;
        for (row_rle.row.items) |item| {
            length_sum += item.length;
            for (0..item.length) |_| {
                rle_back_and_forth.data[y * rle_back_and_forth.width + x] = item.val;
                x += 1;
            }
        }

        const row_start = y * debug.width;
        const row_end = (y + 1) * debug.width;
        var bcs = BarcodeScanner2 {
            .debug = debug.data[row_start..row_end],
            .row = row_rle.row.items,
        };
        _ = bcs.findStartSequence();
        std.debug.assert(x == rle_back_and_forth.width);
    }

    try writeEdges(image.inner, edges, "edges.ppm");
    try writePpmu8(rle_back_and_forth, "rle.ppm");
    try writeDebug(rle_back_and_forth, debug);

    //std.debug.print("suppression\n", .{});
    //const suppressed = try Image(Gradient).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
    //for (0..suppressed.calcHeight()) |y| {
    //    for (0..suppressed.width) |x| {
    //        const in_grad = gradient.data[y * gradient.width + x];
    //        const in_dir = SnappedGradDir.fromAngle(in_grad.direction);
    //        var sample_coords: [2]struct {usize, usize } = switch (in_dir) {
    //            .right => .{
    //                .{x -| 1, y},
    //                .{x + 1, y},
    //            },
    //            .up => .{
    //                .{x , y -| 1},
    //                .{x, y + 1},
    //            },
    //            .up_left => .{
    //                .{x -| 1 , y -| 1},
    //                .{x + 1, y + 1},
    //            },
    //            .up_right => .{
    //                .{x + 1 , y -| 1},
    //                .{x -| 1, y + 1},
    //            },
    //        };

    //        suppressed.data[y * gradient.width + x] = in_grad;

    //        for (&sample_coords) |*sample_coord| {
    //            sample_coord[0] = @min(gradient.width - 1, sample_coord[0]);
    //            sample_coord[1] = @min(gradient.calcHeight() - 1, sample_coord[1]);

    //            const compare = gradient.data[sample_coord[1] * gradient.width + sample_coord[0]];
    //            const compare_dir = SnappedGradDir.fromAngle(compare.direction);
    //            if (compare_dir == in_dir and compare.magnitude > in_grad.magnitude) {
    //                suppressed.data[y * gradient.width + x].magnitude = 0;
    //                break;
    //            }
    //        }

    //        if (in_grad.magnitude < lower_thresh) {
    //            // FIXME: so much indexing
    //            suppressed.data[y * gradient.width + x].confidence = .no;
    //        } else if (in_grad.magnitude < higher_thresh) {
    //            suppressed.data[y * gradient.width + x].confidence = .maybe;
    //        } else {
    //            suppressed.data[y * gradient.width + x].confidence = .yes;
    //        }
    //    }
    //}

    //for (0..suppressed.calcHeight()) |y| {
    //    for (0..suppressed.width) |x| {
    //        var y_it  = y -| 1;
    //        while (y_it <= @min(y + 1, suppressed.calcHeight() - 1)) {
    //            defer y_it += 1;

    //            var x_it  = x -| 1;
    //            while (x_it <= @min(x + 1, suppressed.width - 1)) {
    //                defer x_it += 1;

    //                if (suppressed.data[y_it * suppressed.width + x_it].confidence == .yes) {
    //                    // FIXME: technically we are progressing more than 1 pixel
    //                    suppressed.data[y * suppressed.width + x].confidence = .yes;
    //                }
    //            }
    //        }
    //    }
    //}

    //std.debug.print("Writing to file\n", .{});
    //try writePpmGrad(image.inner, gradient, "out2.ppm");
    //try writePpmGrad(image.inner, suppressed, "out3.ppm");

    //if (false) {
    //    const debug = try Image(DebugPurpose).init(arena.allocator(), image.inner.width, image.inner.calcHeight());
    //    @memset(debug.data, .none);

    //    const processor = try BarcodePreprocessor.init(arena.allocator(), image.inner);
    //    try processor.process();

    //    const scanner = BarcodeScanner {
    //        .image = processor.out,
    //        .debug = debug,
    //    };

    //    for (0..image.inner.calcHeight()) |y_us| {
    //        const y: u31 = @intCast(y_us);
    //        _ = scanner.findStart(y);
    //        _ = scanner.parseLeft(y);
    //    }

    //    try writePpm(processor.out, debug);
    //}
}
