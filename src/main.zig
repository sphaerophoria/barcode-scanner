const std = @import("std");
const stbi = @cImport({
    @cInclude("stb_image.h");
});
const sphtud = @import("sphtud");

const StbiImage = struct {
    inner: Image(u8),

    fn deinit(self: StbiImage) void {
        stbi.stbi_image_free(self.inner.data.ptr);
    }
};

const target_width = 256;

const ImageSize = struct {
    width: u31,
    height: u31,
};

fn getTargetSize(in: Image(u8)) ImageSize {
    const in_height = in.calcHeight();
    if (in_height > in.width) {
        return .{
            .width = in.width * target_width / in_height,
            .height = target_width,
        };
    } else {
        return .{
            .width = target_width,
            .height = in_height * target_width / in.width,
        };
    }
}

fn initialResize(alloc: std.mem.Allocator, in: Image(u8)) !Image(u8) {
    const target_size = getTargetSize(in);
    std.debug.print("{any}\n", .{target_size});
    const ret = try Image(u8).init(alloc, target_size.width, target_size.height);

    const in_height = in.calcHeight();

    var it = ret.iter();
    while (it.nextPixel()) {
        it.pixel().* = in.pixel(
            it.x * in.width / target_size.width,
            it.y * in_height / target_size.height,
        ).*;
    }

    return ret;
}

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
            .data = data[0 .. width * height],
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

pub fn PixelIter(comptime T: type) type {
    return struct {
        // Actual positions in the image
        x: u31,
        y: u31,

        // Logical positions according to iteration. e.g. iterate from [-2, 2].
        // it_x: [-2, -1, 0, 1, 2 ]
        // x:    [0, 0, 0, 1, 2]
        it_x: i32,
        it_y: i32,

        start_x: i32,
        start_y: i32,

        end_x: i32,
        end_y: i32,

        image: Image(T),

        const Self = @This();

        fn init(image: Image(T), start_x: i32, end_x: i32, start_y: i32, end_y: i32) Self {
            var ret = Self{
                // updatePublic...() below
                .x = undefined,
                .y = undefined,
                // NOTE: Expected that nextRow() and nextCol() are called
                // at the start of a loop iterator style
                .it_x = start_x - 1,
                .it_y = start_y - 1,

                .image = image,

                .start_x = start_x,
                .start_y = start_y,

                .end_x = end_x,
                .end_y = end_y,
            };

            ret.updateImageX();
            ret.updateImageY();

            return ret;
        }

        fn nextRow(self: *Self) bool {
            if (self.it_y + 1 >= self.end_y) return false;

            self.it_y += 1;
            // NOTE: Expected that nextCol() is called
            // at the start of a loop iterator style
            self.it_x = self.start_x - 1;

            self.updateImageX();
            self.updateImageY();

            return true;
        }

        fn nextPixel(self: *Self) bool {
            if (self.nextCol()) return true;

            if (self.nextRow()) {
                return self.nextCol();
            }

            return false;
        }

        fn nextCol(self: *Self) bool {
            if (self.it_x + 1 >= self.end_x) return false;
            if (self.it_y < self.start_y) {
                if (self.nextRow() == false) return false;
            }

            self.it_x += 1;
            self.updateImageX();
            return true;
        }

        fn pixel(self: Self) *T {
            return self.image.pixel(self.x, self.y);
        }

        fn updateImageX(self: *Self) void {
            self.x = @intCast(std.math.clamp(self.it_x, 0, self.image.width - 1));
        }

        fn updateImageY(self: *Self) void {
            self.y = @intCast(std.math.clamp(self.it_y, 0, self.image.calcHeight() - 1));
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

        fn iter(self: Self) PixelIter(T) {
            return PixelIter(T).init(
                self,
                0,
                self.width,
                0,
                self.calcHeight(),
            );
        }

        fn areaIt(self: Self, center_x: u31, center_y: u31, width: u31, height: u31) PixelIter(T) {
            std.debug.assert(width % 2 == 1);
            std.debug.assert(height % 2 == 1);

            const half_width = width / 2;
            const half_height = height / 2;

            const start_x = @as(i32, center_x) - half_width;
            const start_y = @as(i32, center_y) - half_height;
            const end_x = center_x + half_width + 1;
            const end_y = center_y + half_height + 1;

            return PixelIter(T).init(
                self,
                start_x,
                end_x,
                start_y,
                end_y,
            );
        }

        fn areaItStartEnd(self: Self, start_x: u31, end_x: u31, start_y: u31, end_y: u31) PixelIter(T) {
            return PixelIter(T).init(
                self,
                start_x,
                end_x,
                start_y,
                end_y,
            );
        }

        fn pixel(self: Self, x: u31, y: u31) *T {
            return &self.data[y * self.width + x];
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

            var it = self.areaIt(x, y, kernel_width, kernel_height);
            while (it.nextRow()) {
                const kernel_y: usize = @intCast(it.it_y - it.start_y);
                while (it.nextCol()) {
                    const kernel_x: usize = @intCast(it.it_x - it.start_x);
                    const mul = kernel[kernel_y * kernel_width + kernel_x];
                    const source = it.pixel().*;
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

        fn applyKernelToImage(self: Self, comptime OutT: type, alloc: std.mem.Allocator, kernel: anytype, post_op: anytype) !Image(OutT) {
            var ret = try Image(OutT).init(alloc, self.width, self.calcHeight());

            var it = self.iter();
            while (it.nextRow()) {
                while (it.nextCol()) {
                    const out_vals = self.applyKernelAtLoc(kernel, it.x, it.y);
                    ret.pixel(it.x, it.y).* = post_op.toPixel(out_vals);
                }
            }

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
                    // FIXME: Check that end module width is ~= start module width
                    // FIXME: Constrain search to that module width
                    if (err_percent < 20) {
                        std.debug.print("err: {d} rle_width: {d}\n", .{ err_percent, end.rle_idx - start.rle_idx });
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

    fn consumeDark(self: BarcodeScanner2, rle_idx: *usize) void {
        if (rle_idx.* >= self.row.len) return;
        const start_x = countRleSizePx(self.row[0..rle_idx.*]);
        var last_val = self.row[rle_idx.*].val;
        while (rle_idx.* < self.row.len and self.row[rle_idx.*].val <= last_val) {
            last_val = self.row[rle_idx.*].val;
            rle_idx.* += 1;
        }
        const end_x = countRleSizePx(self.row[0..rle_idx.*]);
        for (self.debug[start_x..end_x]) |*di| {
            di.* = .dark;
        }
    }

    // FIXME: Heavily duplicated with consumeDark
    fn consumeLight(self: BarcodeScanner2, rle_idx: *usize) void {
        if (rle_idx.* >= self.row.len) return;
        const start_x = countRleSizePx(self.row[0..rle_idx.*]);
        var last_val = self.row[rle_idx.*].val;
        while (rle_idx.* < self.row.len and self.row[rle_idx.*].val >= last_val) {
            last_val = self.row[rle_idx.*].val;
            rle_idx.* += 1;
        }
        const end_x = countRleSizePx(self.row[0..rle_idx.*]);
        for (self.debug[start_x..end_x]) |*di| {
            di.* = .light;
        }
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
        .{ 3, 2, 1, 1 },
        .{ 2, 2, 2, 1 },
        .{ 2, 1, 2, 2 },
        .{ 1, 4, 1, 1 },
        .{ 1, 1, 3, 2 },
        .{ 1, 2, 3, 1 },
        .{ 1, 1, 1, 4 },
        .{ 1, 3, 1, 2 },
        .{ 1, 2, 1, 3 },
        .{ 3, 1, 1, 2 },
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

    fn findStart(self: BarcodeScanner, y: u31) ?struct { u31, u31 } {
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
                    return .{ out_x, module_width };
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
    dark,
    light,
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
    , .{ image.width, image.calcHeight() });

    for (0..image.calcHeight()) |y| {
        for (0..image.width) |x| {
            const to_write = switch (debug.data[y * image.width + x]) {
                .none => blk: {
                    const val: u8 = if (image.data[y * image.width + x] > 0) 255 else 0;
                    break :blk .{ val, val, val };
                },
                .potential_barcode_start => .{ 255, 255, 0 },
                .barcode_start => .{ 0, 0, 255 },
                .barcode_end => .{ 255, 0, 0 },
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
    , .{ image.width, image.calcHeight() });

    for (0..image.calcHeight()) |y| {
        for (0..image.width) |x| {
            const to_write = switch (debug.data[y * image.width + x]) {
                .none => blk: {
                    const val: u8 = image.data[y * image.width + x];
                    break :blk .{ val, val, val };
                },
                .potential_barcode_start => .{ 255, 255, 0 },
                .barcode_start => .{ 0, 0, 255 },
                .barcode_end => .{ 255, 0, 0 },
                .dark => .{ 0, 0, 0 },
                .light => .{ 255, 255, 255 },
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
    , .{ image.width, image.calcHeight() });

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
    , .{ image.width, image.calcHeight() });

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

const Rgb = struct {
    r: f32,
    g: f32,
    b: f32,
};

fn hsvToRgb(h: f32, s: f32, v: f32) Rgb {
    var r_temp: f32 = undefined;
    var g_temp: f32 = undefined;
    var b_temp: f32 = undefined;
    std.debug.assert(h >= 0 and h <= std.math.pi);

    if (s == 0) {
        r_temp = v;
        g_temp = v;
        b_temp = v;
    } else {
        const sector = h / std.math.pi * 3; // sector 0 to 5
        const i: i32 = @intFromFloat(sector); // integer part of sector
        const f = sector - @as(f32, @floatFromInt(i)); // fractional part of sector
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t = v * (1 - (1 - f) * s);

        switch (i) {
            0 => {
                r_temp = v;
                g_temp = t;
                b_temp = p;
            },
            1 => {
                r_temp = q;
                g_temp = v;
                b_temp = p;
            },
            2 => {
                r_temp = p;
                g_temp = v;
                b_temp = t;
            },
            3 => {
                r_temp = p;
                g_temp = q;
                b_temp = v;
            },
            4 => {
                r_temp = t;
                g_temp = p;
                b_temp = v;
            },
            else => {
                r_temp = v;
                g_temp = p;
                b_temp = q;
            },
        }
    }

    // Convert to 0-255 range
    return .{
        .r = r_temp * 255,
        .g = g_temp * 255,
        .b = b_temp * 255,
    };
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
    , .{ image.width, image.calcHeight() });

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
            const mag = val.length2();
            const color = hsvToRgb(val.direction, 1, 1);
            try ppm_writer.writeByte(@intFromFloat(color.r * mag));
            try ppm_writer.writeByte(@intFromFloat(color.g * mag));
            try ppm_writer.writeByte(@intFromFloat(color.b * mag));
        }
    }
}

fn gradientToTexture(gl_alloc: *sphtud.render.GlAlloc, scratch: *sphtud.alloc.ScratchAlloc, image: Image(Gradient)) !sphtud.render.Texture {
    const checkpoint = scratch.checkpoint();
    defer scratch.restore(checkpoint);

    const out = try Image(GlPixel).init(scratch.allocator(), image.width, image.calcHeight());
    var it = image.iter();
    while (it.nextPixel()) {
        const val = it.pixel().*;
        const pixel: GlPixel = switch (val.confidence) {
            .no, .maybe => {
                unreachable;
            },
            else => blk: {
                break :blk .{
                    .r = @intFromFloat(@abs(val.v[0] * 255)),
                    .g = @intFromFloat(@abs(val.v[1] * 255)),
                    .b = 0,
                    .a = 255,
                };
            },
        };

        out.pixel(it.x, it.y).* = pixel;
    }

    return try sphtud.render.makeTextureFromRgba(gl_alloc, @ptrCast(out.data), image.width);
}

fn writeClusters(clusters: Clusters, image: Image(Gradient), out_path: []const u8) !void {
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
    , .{ image.width, image.calcHeight() });

    std.debug.print("num clusters {d}\n", .{clusters.num_clusters});
    var it = image.iter();
    while (it.nextRow()) {
        while (it.nextCol()) {
            const val = it.pixel().*;
            const mag = val.length2();

            const cluster_id = clusters.image.pixel(it.x, it.y).*;

            if (cluster_id == 0) {
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
                continue;
            }

            var cluster_angle: f32 = @floatFromInt((cluster_id * 1) % clusters.num_clusters);
            cluster_angle *= std.math.pi;
            cluster_angle /= @floatFromInt(clusters.num_clusters);

            const color = hsvToRgb(cluster_angle, 1, 1);
            if (false) {
                try ppm_writer.writeByte(@intFromFloat(color.r * mag));
                try ppm_writer.writeByte(@intFromFloat(color.g * mag));
                try ppm_writer.writeByte(@intFromFloat(color.b * mag));
            }
            try ppm_writer.writeByte(@intFromFloat(color.r));
            try ppm_writer.writeByte(@intFromFloat(color.g));
            try ppm_writer.writeByte(@intFromFloat(color.b));
        }
    }
}

fn clustersToTex(gl_alloc: *sphtud.render.GlAlloc, scratch: *sphtud.alloc.ScratchAlloc, clusters: Clusters, gradient: Image(Gradient), selected_point: ?Point2D) !sphtud.render.Texture {
    const checkpoint = scratch.checkpoint();
    defer scratch.restore(checkpoint);

    std.debug.assert(gradient.width == clusters.image.width);
    std.debug.assert(gradient.calcHeight() == clusters.image.calcHeight());

    const out = try Image(GlPixel).init(scratch.allocator(), gradient.width, gradient.calcHeight());
    @memset(out.data, .{ .r = 0, .g = 0, .b = 0, .a = 255 });

    var it = gradient.iter();
    while (it.nextRow()) {
        while (it.nextCol()) {
            const val = it.pixel().*;
            var mag = sphtud.math.length2(val.v);
            const cluster_id = clusters.image.pixel(it.x, it.y).*;

            if (cluster_id == 0) {
                continue;
            }

            var cluster_angle: f32 = @floatFromInt((cluster_id * 1) % clusters.num_clusters);
            cluster_angle *= std.math.pi;
            cluster_angle /= @floatFromInt(clusters.num_clusters);

            if (true) mag = 1.0;
            var color = hsvToRgb(cluster_angle, 1, 1);

            if (selected_point) |sp| {
                if (sp.x == it.x and sp.y == it.y) {
                    color = .{
                        .r = 0,
                        .g = 255,
                        .b = 255,
                    };
                }
            }

            out.pixel(it.x, it.y).* = .{
                .r = @intFromFloat(color.r * mag),
                .g = @intFromFloat(color.g * mag),
                .b = @intFromFloat(color.b * mag),
                .a = 255,
            };
        }
    }

    return sphtud.render.makeTextureFromRgba(gl_alloc, @ptrCast(out.data), out.width);
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
    , .{ edges.width, edges.calcHeight() });

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

fn writeEdgePass1(original: Image(u8), edges: Image(i32), out_path: []const u8) !void {
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
    , .{ edges.width, edges.calcHeight() });

    for (0..edges.calcHeight()) |y| {
        for (0..edges.width) |x| {
            const val = edges.data[y * edges.width + x];
            if (val == 0) {
                const source_val = original.data[y * original.width + x];
                try ppm_writer.writeByte(source_val);
                try ppm_writer.writeByte(source_val);
                try ppm_writer.writeByte(source_val);
            } else {
                try ppm_writer.writeByte(@intCast(@min(@abs(val), 255)));
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
            }
        }
    }
}

const Gradient = struct {
    // radians[0, 2pi],
    //direction: f32,
    //magnitude: f32,
    v: sphtud.math.Vec2,
    confidence: enum { no, maybe, yes } = .yes,

    fn angle(self: Gradient) f32 {
        return std.math.atan2(self.v[1], self.v[0]);
    }
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

            return @divTrunc(ret, @as(T, @intCast(self.used)));
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

const horizontal_gradient_kernel = StackMat2D(i32, 3, 3){
    .data = .{
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1,
    },
};

const vertical_gradient_kernel = StackMat2D(i32, 3, 3){
    .data = .{
        -1, -2, -1,
        0,  0,  0,
        1,  2,  1,
    },
};

fn edgePass1(alloc: std.mem.Allocator, blurry: Image(u8)) !Image(i32) {
    // Find edges
    const edges = try Image(i32).init(alloc, blurry.width, blurry.calcHeight());
    @memset(edges.data, 0);

    var edges_it = edges.iter();
    std.debug.print("Calculating gradient\n", .{});

    var gradient_average = RollingAverage(i32, 1){};
    const gradient_thresh = 20;

    while (edges_it.nextRow()) {
        var last_average: i32 = 0;
        var increasing: bool = true;

        while (edges_it.nextCol()) {
            const this_gradient = blurry.applyKernelAtLoc(horizontal_gradient_kernel, edges_it.x, edges_it.y).sum();
            gradient_average.push(this_gradient);
            const saveraged = gradient_average.average();
            const abs_averaged = @abs(saveraged);
            const last_abs_average = @abs(last_average);

            if (abs_averaged < last_abs_average and increasing and last_abs_average > gradient_thresh) {
                edges_it.pixel().* = last_average;
            }

            if (abs_averaged <= last_abs_average) {
                increasing = false;
            } else {
                increasing = true;
            }

            last_average = saveraged;
        }
    }
    return edges;
}

fn hasHorizontalSibling(edges: Image(i32), x: u31, y: u31) bool {
    var it = edges.areaIt(x, y, 5, 1);
    while (it.nextCol()) {
        if (edges.pixel(it.x, it.y).* != 0) {
            return true;
        }
    }
    return false;
}

fn getNearbyGradient(edges: Image(i32), x: u31, y: u31) ?i32 {
    // Looking for 2 rows up and down for pixel directly to the or right to
    // merge with above/below edges
    var it = edges.areaIt(x, y, 1, 5);
    while (it.nextRow()) {
        if (it.y == y) continue;

        while (it.nextCol()) {
            const gradient = it.pixel().*;
            if (gradient != 0) {
                // FIXME: Shouldn't just pick the first one
                return gradient;
            }
        }
    }

    return null;
}

const SobelPostOp = struct {
    pub fn toPixel(in: StackMat2D(i32, 3, 3)) u8 {
        return @intCast(@abs(in.sum()) / 4);
    }
};

fn imageGradient(alloc: std.mem.Allocator, input: Image(u8)) !Image(Gradient) {
    //const gaussian_blur_kernel = StackMat2D(f32, 5, 5) {
    //    .data = .{
    //        1, 4, 6, 4, 1,
    //        4, 16, 24, 16, 4,
    //        6, 24, 36, 24, 6,
    //        4, 16, 24, 16, 4,
    //        1, 4, 6, 4, 1,
    //    },
    //};

    //const BlurPostOp = struct {
    //    pub fn toPixel(in: anytype) f32 {
    //        return in.sum() / 256;
    //    }

    //};

    const out = try Image(Gradient).init(alloc, input.width, input.calcHeight());

    var it = input.iter();
    while (it.nextRow()) {
        while (it.nextCol()) {
            const horizontal_grad = input.applyKernelAtLoc(horizontal_gradient_kernel, it.x, it.y).sum();
            const vertical_grad = input.applyKernelAtLoc(vertical_gradient_kernel, it.x, it.y).sum();
            var hf: f32 = @floatFromInt(horizontal_grad);
            var vf: f32 = @floatFromInt(vertical_grad);
            // Normalize [0,1]
            hf /= 255 * 4;
            vf /= 255 * 4;

            if (vf < 0) {
                hf = -hf;
                vf = -vf;
            }

            out.pixel(it.x, it.y).* = .{
                .v = .{ hf, vf },
            };
        }
    }

    return out;
}

const Point2D = struct {
    x: u31,
    y: u31,
};

const Clusters = struct {
    num_clusters: usize,
    image: Image(usize),
};

fn Timer(comptime max_elems: comptime_int) type {
    return struct {
        times: std.BoundedArray(Item, max_elems) = .{},

        const Item = struct {
            name: []const u8,
            time: std.time.Instant,
        };

        const Self = @This();

        fn init() !Self {
            var ret = Self{};
            try ret.markStart("start");
            return ret;
        }

        fn markStart(self: *Self, name: []const u8) !void {
            const now = try std.time.Instant.now();
            try self.times.append(.{
                .time = now,
                .name = name,
            });
        }

        fn log(self: *Self) !void {
            try self.markStart("end");
            const times = self.times.slice();

            for (0..times.len - 1) |i| {
                const start = times[i];
                const end = times[i + 1];

                const elapsed_ms = end.time.since(start.time) / std.time.ns_per_ms;
                std.debug.print("{s} took {d}ms\n", .{ start.name, elapsed_ms });
            }
        }
    };
}

const Clusterer = struct {
    alloc: std.mem.Allocator,
    cluster_list: std.AutoHashMap(usize, std.AutoHashMap(Point2D, void)),
    out: Image(usize),
    cluster_id_cache: std.ArrayList(usize),
    biggest_cluster_id: usize,
    input: Image(Gradient),
    it: PixelIter(Gradient),
    num_pixels_visited: usize = 0,
    min_cluster_mag: f32,
    min_cluster_similarity: f32,

    pub fn init(alloc: std.mem.Allocator, input: Image(Gradient), min_cluster_mag: f32, min_cluster_similarity: f32) !Clusterer {
        const out = try Image(usize).init(alloc, input.width, input.calcHeight());
        @memset(out.data, 0);

        return .{
            .alloc = alloc,
            .cluster_list = std.AutoHashMap(usize, std.AutoHashMap(Point2D, void)).init(alloc),
            .input = input,
            .it = input.iter(),
            .cluster_id_cache = std.ArrayList(usize).init(alloc),
            .biggest_cluster_id = 1,
            .out = out,
            .num_pixels_visited = 0,
            .min_cluster_mag = min_cluster_mag,
            .min_cluster_similarity = min_cluster_similarity,
        };
    }

    fn step(self: *Clusterer) !bool {
        if (!self.it.nextPixel()) return false;

        // Every pixel in its own cluster
        // Find closest pair
        //   * sort by pixel distance + thresholded angle difference
        //   * sort by angle difference
        self.num_pixels_visited += 1;
        const our_gradient = self.it.pixel().*;
        const our_cluster = self.out.pixel(self.it.x, self.it.y);

        if (sphtud.math.length2(our_gradient.v) < self.min_cluster_mag) return true;
        var cluster_area_it = self.input.areaItStartEnd(self.it.x, self.it.x + 2, self.it.y, self.it.y + 2);
        while (cluster_area_it.nextPixel()) {
            self.num_pixels_visited += 1;

            const their_gradient = cluster_area_it.pixel().*;
            if (sphtud.math.length2(their_gradient.v) < self.min_cluster_mag) continue;
            // FIXME: if 89 degrees we should consider those close
            const similarity = @abs(sphtud.math.dot(
                sphtud.math.normalize(our_gradient.v),
                sphtud.math.normalize(their_gradient.v),
            ));

            if (similarity > self.min_cluster_similarity) {
                const their_cluster = self.out.pixel(cluster_area_it.x, cluster_area_it.y);

                if (our_cluster.* != 0 and their_cluster.* != 0 and our_cluster.* != their_cluster.*) {
                    const original_their_cluster = their_cluster.*;
                    // Could merge, but we are looking for barcodes
                    // which are square... so we shouldn't need to
                    // merge disjoint clusters. The geometry should
                    // prevent any relevant case of this happening
                    const their_cluster_pixels = self.cluster_list.getPtr(original_their_cluster);
                    const our_cluster_pixels = self.cluster_list.getPtr(our_cluster.*);

                    var their_pixel_it = their_cluster_pixels.?.keyIterator();
                    while (their_pixel_it.next()) |pixel| {
                        self.num_pixels_visited += 1;
                        std.debug.assert(self.out.pixel(pixel.x, pixel.y).* == original_their_cluster);
                        self.out.pixel(pixel.x, pixel.y).* = our_cluster.*;
                        try our_cluster_pixels.?.put(pixel.*, {});
                    }
                    _ = self.cluster_list.remove(original_their_cluster);
                    try self.cluster_id_cache.append(original_their_cluster);
                } else if (their_cluster.* == 0 and our_cluster.* == 0) {
                    const new_cluster_id = if (self.cluster_id_cache.pop()) |v| v else blk: {
                        defer self.biggest_cluster_id += 1;
                        break :blk self.biggest_cluster_id;
                    };
                    their_cluster.* = new_cluster_id;
                    our_cluster.* = new_cluster_id;
                    var new_list = std.AutoHashMap(Point2D, void).init(self.alloc);
                    try new_list.put(.{
                        .x = self.it.x,
                        .y = self.it.y,
                    }, {});
                    try new_list.put(.{
                        .x = cluster_area_it.x,
                        .y = cluster_area_it.y,
                    }, {});
                    try self.cluster_list.put(new_cluster_id, new_list);
                } else if (their_cluster.* == 0) {
                    const l = self.cluster_list.getPtr(our_cluster.*);
                    try l.?.put(.{
                        .x = cluster_area_it.x,
                        .y = cluster_area_it.y,
                    }, {});
                    their_cluster.* = our_cluster.*;
                } else if (our_cluster.* == 0) {
                    const l = self.cluster_list.getPtr(their_cluster.*);
                    try l.?.put(.{
                        .x = self.it.x,
                        .y = self.it.y,
                    }, {});
                    our_cluster.* = their_cluster.*;
                }
            }
        }

        return true;
    }
};

fn downsampleImage(alloc: std.mem.Allocator, input: anytype, divisor: u31) !@TypeOf(input) {
    const T = @TypeOf(input);
    var ret = try T.init(alloc, input.width / divisor, input.calcHeight() / divisor);
    @memset(ret.data, .{
        .v = .{ 0, 0 },
    });

    var it = ret.iter();
    while (it.nextPixel()) {
        var input_it = input.areaItStartEnd(it.x * divisor, (it.x + 1) * divisor, it.y * divisor, (it.y + 1) * divisor);
        var max: Gradient = .{ .v = .{ 0, 0 } };
        while (input_it.nextPixel()) {
            if (sphtud.math.length2(input_it.pixel().v) > sphtud.math.length2(max.v)) {
                max = input_it.pixel().*;
            }
        }
        it.pixel().* = max;
    }

    return ret;
}

fn clusterImageGradient(alloc: std.mem.Allocator, input: Image(Gradient), min_cluster_mag: f32, min_cluster_similarity: f32, min_cluster_size: usize) !Clusters {
    var timer = try Timer(10).init();

    try timer.markStart("clustering");
    var clusterer = try Clusterer.init(alloc, input, min_cluster_mag, min_cluster_similarity);

    while (try clusterer.step()) {}

    std.debug.print("Num pixels visisted: {d}\n", .{clusterer.num_pixels_visited});

    try timer.markStart("Thresholding");

    var removed_cluster_num: usize = 0;
    for (0..clusterer.biggest_cluster_id) |id| {
        if (id == 0) continue;
        const pixel_list = clusterer.cluster_list.get(id) orelse continue;
        var pixel_it = pixel_list.keyIterator();
        while (pixel_it.next()) |pixel| {
            std.debug.assert(clusterer.out.pixel(pixel.x, pixel.y).* == id);
        }
        if (pixel_list.count() < min_cluster_size) {
            removed_cluster_num += 1;
            pixel_it = pixel_list.keyIterator();
            while (pixel_it.next()) |pixel| {
                clusterer.out.pixel(pixel.x, pixel.y).* = 0;
            }
            _ = clusterer.cluster_list.remove(id);
        }
    }

    //std.debug.print("Found {d} clusters", .{cluster_list.count()});

    //var cluster_it = cluster_list.iterator();
    //while (cluster_it.next()) |entry| {
    //    var point_it = entry.value_ptr.keyIterator();
    //    while (point_it.next()) |point| {
    //        const dir = input.pixel(point.x, point.y).direction;
    //    }
    //}

    try timer.log();
    return .{
        .num_clusters = clusterer.biggest_cluster_id,
        .image = clusterer.out,
    };
}

const GuiAction = union(enum) {
    update_cluster_mag: f32,
    update_cluster_angle: f32,
    update_step_size: f32,
    step_cluster,
    reset_cluster,

    fn updateStepSize(val: f32) GuiAction {
        return .{ .update_step_size = val };
    }

    fn updateClusterMag(val: f32) GuiAction {
        return .{ .update_cluster_mag = val };
    }

    fn updateClusterAngle(val: f32) GuiAction {
        return .{ .update_cluster_angle = val };
    }
};

const ImageTexture = struct {
    tex: sphtud.render.Texture,
    width: u31,
    height: u31,

    fn init(gl_alloc: *sphtud.render.GlAlloc, image: Image(u8)) !ImageTexture {
        const tex = try sphtud.render.makeTextureFromR(gl_alloc, image.data, image.width);
        return .{
            .tex = tex,
            .width = image.width,
            .height = image.calcHeight(),
        };
    }

    pub fn getTexture(self: ImageTexture) sphtud.render.Texture {
        return self.tex;
    }

    pub fn getSize(self: ImageTexture) sphtud.ui.PixelSize {
        return .{
            .width = self.width,
            .height = self.height,
        };
    }
};

const ImageView = struct {
    image_renderer: *const sphtud.render.xyuvt_program.ImageRenderer,
    tex: sphtud.render.Texture,
    width: u31,
    aspect: f32,

    fn init(
        gui_alloc: sphtud.ui.GuiAlloc,
        image_renderer: *const sphtud.render.xyuvt_program.ImageRenderer,
        image: Image(u8),
    ) !sphtud.ui.Widget(GuiAction) {
        const tex = try sphtud.render.makeTextureFromR(gui_alloc.gl, image.data, image.width);
        const ret = try gui_alloc.heap.arena().create(ImageView);
        const aspect = @as(f32, @floatFromInt(image.width)) / @as(f32, @floatFromInt(image.calcHeight()));
        ret.* = .{
            .image_renderer = image_renderer,
            .tex = tex,
            .width = image.width,
            .aspect = aspect,
        };
        return ret.asWidget();
    }

    fn initTex(
        gui_alloc: sphtud.ui.GuiAlloc,
        image_renderer: *const sphtud.render.xyuvt_program.ImageRenderer,
        tex: sphtud.render.Texture,
        width: u31,
        height: u31,
    ) !sphtud.ui.Widget(GuiAction) {
        const ret = try gui_alloc.heap.arena().create(ImageView);
        const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
        ret.* = .{
            .image_renderer = image_renderer,
            .tex = tex,
            .width = width,
            .aspect = aspect,
        };
        return ret.asWidget();
    }

    fn initEmpty(
        image_renderer: *const sphtud.render.xyuvt_program.ImageRenderer,
    ) !ImageView {
        return .{
            .image_renderer = image_renderer,
            .tex = .invalid,
            .width = 0,
            .aspect = 1.0,
        };
    }

    const vtable = sphtud.ui.Widget(GuiAction).VTable{
        .render = render,
        .getSize = getSize,
        .update = update,
        .setInputState = null,
        .setFocused = null,
        .reset = null,
    };

    fn asWidget(self: *ImageView) sphtud.ui.Widget(GuiAction) {
        return .{
            .vtable = &vtable,
            .name = "image_view",
            .ctx = self,
        };
    }

    fn render(ctx: ?*anyopaque, widget_bounds: sphtud.ui.PixelBBox, window_bounds: sphtud.ui.PixelBBox) void {
        const self: *ImageView = @ptrCast(@alignCast(ctx));
        const transform = sphtud.ui.util.widgetToClipTransform(widget_bounds, window_bounds);
        self.image_renderer.renderTexture(self.tex, sphtud.math.Transform.scale(1, -1).then(transform));
    }

    fn getSize(ctx: ?*anyopaque) sphtud.ui.PixelSize {
        const self: *ImageView = @ptrCast(@alignCast(ctx));
        return .{
            .width = self.width,
            .height = @intFromFloat(@as(f32, @floatFromInt(self.width)) / self.aspect),
        };
    }

    fn update(ctx: ?*anyopaque, available_size: sphtud.ui.PixelSize, _: f32) anyerror!void {
        const self: *ImageView = @ptrCast(@alignCast(ctx));

        self.width = available_size.width;
    }
};

const GlPixel = packed struct(u32) {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

fn debugToTexture(out_alloc: *sphtud.render.GlAlloc, scratch: *sphtud.alloc.ScratchAlloc, image: Image(u8), debug: Image(DebugPurpose)) !sphtud.render.Texture {
    const checkpoint = scratch.checkpoint();
    defer scratch.restore(checkpoint);

    std.debug.assert(image.width == debug.width);
    std.debug.assert(image.calcHeight() == debug.calcHeight());

    const out = try Image(GlPixel).init(scratch.allocator(), image.width, image.calcHeight());

    var it = debug.iter();
    while (it.nextPixel()) {
        const pixel: GlPixel = switch (debug.pixel(it.x, it.y).*) {
            .none => blk: {
                const val: u8 = image.pixel(it.x, it.y).*;
                break :blk .{ .r = val, .g = val, .b = val, .a = 255 };
            },
            .potential_barcode_start => .{ .r = 255, .g = 255, .b = 0, .a = 255 },
            .barcode_start => .{ .r = 0, .g = 0, .b = 255, .a = 255 },
            .barcode_end => .{ .r = 255, .g = 0, .b = 0, .a = 255 },
            .dark => .{ .r = 0, .g = 0, .b = 0, .a = 255 },
            .light => .{ .r = 255, .g = 255, .b = 255, .a = 255 },
        };
        out.pixel(it.x, it.y).* = pixel;
    }

    return try sphtud.render.makeTextureFromRgba(out_alloc, @ptrCast(out.data), image.width);
}

fn initGlParams() void {
    sphtud.render.gl.glEnable(sphtud.render.gl.GL_MULTISAMPLE);
    sphtud.render.gl.glEnable(sphtud.render.gl.GL_SCISSOR_TEST);
    sphtud.render.gl.glBlendFunc(sphtud.render.gl.GL_SRC_ALPHA, sphtud.render.gl.GL_ONE_MINUS_SRC_ALPHA);
    sphtud.render.gl.glEnable(sphtud.render.gl.GL_BLEND);
}

const ClusteringResults = struct {
    sobel: Image(Gradient),
    low_res_sobel: Image(Gradient),
    clusters: Clusters,
};

const ImageViews = struct {
    image_alloc: *sphtud.render.GlAlloc,
    sobel: ImageView,
    low_res_sobel: ImageView,
    clusters: ImageView,

    debug_alloc: *sphtud.render.GlAlloc,
    debug_cluster: ImageView,

    fn init(gui_alloc: sphtud.ui.GuiAlloc, rgba_renderer: *const sphtud.render.xyuvt_program.ImageRenderer) !ImageViews {
        const image_alloc = (try gui_alloc.makeSubAlloc("image_textures")).gl;
        const debug_alloc = (try gui_alloc.makeSubAlloc("debug_cluster")).gl;
        return .{
            .image_alloc = image_alloc,
            .sobel = try ImageView.initEmpty(rgba_renderer),
            .low_res_sobel = try ImageView.initEmpty(rgba_renderer),
            .clusters = try ImageView.initEmpty(rgba_renderer),
            .debug_alloc = debug_alloc,
            .debug_cluster = try ImageView.initEmpty(rgba_renderer),
        };
    }

    fn update(self: *ImageViews, scratch: *sphtud.alloc.ScratchAlloc, results: ClusteringResults) !void {
        self.image_alloc.reset();
        errdefer {
            self.sobel.tex = .invalid;
            self.clusters.tex = .invalid;
        }

        self.sobel.tex = try gradientToTexture(self.image_alloc, scratch, results.sobel);
        self.sobel.aspect = @floatFromInt(results.sobel.width);
        self.sobel.aspect /= @floatFromInt(results.sobel.calcHeight());

        self.low_res_sobel.tex = try gradientToTexture(self.image_alloc, scratch, results.low_res_sobel);
        self.low_res_sobel.aspect = @floatFromInt(results.low_res_sobel.width);
        self.low_res_sobel.aspect /= @floatFromInt(results.low_res_sobel.calcHeight());

        self.clusters.tex = try clustersToTex(
            self.image_alloc,
            scratch,
            results.clusters,
            results.low_res_sobel,
            null,
        );
        self.clusters.aspect = @floatFromInt(results.clusters.image.width);
        self.clusters.aspect /= @floatFromInt(results.clusters.image.calcHeight());
    }

    fn updateDebug(self: *ImageViews, scratch: *sphtud.alloc.ScratchAlloc, clusterer: Clusterer) !void {
        self.debug_cluster.tex = try clustersToTex(
            self.debug_alloc,
            scratch,
            .{
                .image = clusterer.out,
                .num_clusters = clusterer.biggest_cluster_id,
            },
            clusterer.input,
            .{ .x = clusterer.it.x, .y = clusterer.it.y },
        );
        self.debug_cluster.aspect = @floatFromInt(clusterer.out.width);
        self.debug_cluster.aspect /= @floatFromInt(clusterer.out.calcHeight());
    }
};

const GuiParams = struct {
    min_cluster_mag: f32 = 0.02,
    min_cluster_similarity: f32 = 0.95,
    step_size: f32 = 1.0,
};

const Gui = struct {
    image_views: *ImageViews,
    runner: sphtud.ui.runner.Runner(GuiAction),
};

fn makeGuiRoot(gui_alloc: sphtud.ui.GuiAlloc, scratch: *sphtud.alloc.ScratchAlloc, gl_scratch: *sphtud.render.GlAlloc, image: Image(u8), gui_params: *const GuiParams) !Gui {
    var gui_state = try sphtud.ui.widget_factory.widgetState(GuiAction, gui_alloc, scratch, gl_scratch);

    const widget_factory = gui_state.factory(gui_alloc);
    const root_layout = try widget_factory.makeLayout();
    const root_widget = try widget_factory.makeScrollView(root_layout.asWidget());

    const runner = try widget_factory.makeRunner(root_widget);

    const grey_image_renderer = try gui_alloc.heap.arena().create(sphtud.render.xyuvt_program.ImageRenderer);
    grey_image_renderer.* = try sphtud.render.xyuvt_program.ImageRenderer.init(gui_alloc.gl, .greyscale);

    const image_views = try gui_alloc.heap.arena().create(ImageViews);
    image_views.* = try ImageViews.init(gui_alloc, &gui_state.image_renderer);

    try root_layout.pushWidget(try widget_factory.makeLabel("Resized initial image"));
    try root_layout.pushWidget(try ImageView.init(gui_alloc, grey_image_renderer, image));

    // FIXME: Attach results to own allocator to nuke on parameter change

    try root_layout.pushWidget(try widget_factory.makeLabel("Sobel"));
    try root_layout.pushWidget(image_views.sobel.asWidget());

    try root_layout.pushWidget(try widget_factory.makeLabel("Sobel low res"));
    try root_layout.pushWidget(image_views.low_res_sobel.asWidget());

    try root_layout.pushWidget(try widget_factory.makeLabel("Clusters"));
    try root_layout.pushWidget(try widget_factory.makeLabel("Min cluster magnitude"));
    try root_layout.pushWidget(try widget_factory.makeDragFloat(
        &gui_params.min_cluster_mag,
        &GuiAction.updateClusterMag,
        0.001,
    ));

    try root_layout.pushWidget(try widget_factory.makeLabel("Max cluster angle"));
    try root_layout.pushWidget(try widget_factory.makeDragFloat(
        &gui_params.min_cluster_similarity,
        &GuiAction.updateClusterAngle,
        0.001,
    ));

    try root_layout.pushWidget(image_views.clusters.asWidget());

    try root_layout.pushWidget(try widget_factory.makeLabel("Debug cluster view"));
    try root_layout.pushWidget(try widget_factory.makeButton("Step", GuiAction{ .step_cluster = {} }));
    try root_layout.pushWidget(try widget_factory.makeButton("Reset", GuiAction{ .reset_cluster = {} }));
    try root_layout.pushWidget(try widget_factory.makeLabel("Step size"));
    try root_layout.pushWidget(try widget_factory.makeDragFloat(
        &gui_params.step_size,
        &GuiAction.updateStepSize,
        0.1,
    ));
    try root_layout.pushWidget(image_views.debug_cluster.asWidget());

    return .{
        .image_views = image_views,
        .runner = runner,
    };
}

const gradient_downsample_factor = 6;
fn makeClusters(scratch: *sphtud.alloc.ScratchAlloc, image: Image(u8), gui_params: GuiParams) !ClusteringResults {
    const sobel_out = try imageGradient(scratch.allocator(), image);

    const sobel_resized = try downsampleImage(scratch.allocator(), sobel_out, gradient_downsample_factor);
    const clusters = try clusterImageGradient(
        scratch.allocator(),
        sobel_resized,
        gui_params.min_cluster_mag,
        gui_params.min_cluster_similarity,
        sobel_resized.width * sobel_resized.calcHeight() / 10,
    );
    return .{
        .sobel = sobel_out,
        .low_res_sobel = sobel_resized,
        .clusters = clusters,
    };
}

fn updateImageViews(scratch: *sphtud.alloc.ScratchAlloc, image: Image(u8), gui_params: GuiParams, image_views: *ImageViews) !void {
    const results = makeClusters(scratch, image, gui_params) catch |e| {
        std.debug.print("{s}\n", .{@errorName(e)});
        std.debug.print("{}\n", .{@errorReturnTrace().?});
        return;
    };
    try image_views.update(scratch, results);
}

pub fn main() !void {
    var window: sphtud.window.Window = undefined;
    try window.initPinned("barcode-scanner", 800, 600);

    var allocators: sphtud.render.AppAllocators(100) = undefined;
    try allocators.initPinned(10 * 1024 * 1024);

    initGlParams();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const path = (try std.process.argsAlloc(arena.allocator()))[1];

    const image = blk: {
        const image = try loadImage(path);
        defer image.deinit();

        break :blk try initialResize(arena.allocator(), image.inner);
    };

    //const gradient = try Image(Gradient).init(arena.allocator(), image.inner.width, image.inner.calcHeight());

    var gui_params = GuiParams{};
    var gui = try makeGuiRoot(try allocators.root_render.makeSubAlloc("gui"), &allocators.scratch, &allocators.scratch_gl, image, &gui_params);
    const runner = &gui.runner;
    const image_views = gui.image_views;

    try updateImageViews(&allocators.scratch, image, gui_params, image_views);

    const debug_cluster_sobel = try downsampleImage(
        allocators.root.arena(),
        try imageGradient(allocators.scratch.allocator(), image),
        gradient_downsample_factor,
    );

    const clusterer_alloc = try allocators.root.makeSubAlloc("clusterer");
    var clusterer = try Clusterer.init(clusterer_alloc.arena(), debug_cluster_sobel, gui_params.min_cluster_mag, gui_params.min_cluster_similarity);

    while (!window.closed()) {
        allocators.resetScratch();

        const window_width, const window_height = window.getWindowSize();

        sphtud.render.gl.glClear(sphtud.render.gl.GL_COLOR_BUFFER_BIT);

        sphtud.render.gl.glViewport(0, 0, @intCast(window_width), @intCast(window_height));
        sphtud.render.gl.glScissor(0, 0, @intCast(window_width), @intCast(window_height));

        const action = try runner.step(1.0, .{
            .width = @intCast(window_width),
            .height = @intCast(window_height),
        }, &window.queue);

        if (action.action) |a| switch (a) {
            .update_cluster_mag => |v| {
                gui_params.min_cluster_mag = @max(v, 0);
                updateImageViews(&allocators.scratch, image, gui_params, image_views) catch {};

                try clusterer_alloc.reset();
                clusterer = try Clusterer.init(clusterer_alloc.arena(), debug_cluster_sobel, gui_params.min_cluster_mag, gui_params.min_cluster_similarity);
            },
            .update_cluster_angle => |v| {
                gui_params.min_cluster_similarity = @max(v, 0);
                updateImageViews(&allocators.scratch, image, gui_params, image_views) catch {};

                try clusterer_alloc.reset();
                clusterer = try Clusterer.init(clusterer_alloc.arena(), debug_cluster_sobel, gui_params.min_cluster_mag, gui_params.min_cluster_similarity);
            },
            .update_step_size => |v| {
                gui_params.step_size = @round(@max(v, 0));
            },
            .reset_cluster => {
                try clusterer_alloc.reset();
                clusterer = try Clusterer.init(clusterer_alloc.arena(), debug_cluster_sobel, gui_params.min_cluster_mag, gui_params.min_cluster_similarity);
                for (0..1100) |_| {
                    _ = try clusterer.step();
                }
                try image_views.updateDebug(&allocators.scratch, clusterer);
            },
            .step_cluster => {
                const step_size: usize = @intFromFloat(gui_params.step_size);
                for (0..step_size) |_| {
                    _ = try clusterer.step();
                }
                try image_views.updateDebug(&allocators.scratch, clusterer);
            },
        };

        window.swapBuffers();
    }

    const blur_kernel = StackMat2D(u16, 1, 5){
        .data = .{
            5,
            12,
            15,
            12,
            5,
        },
    };

    std.debug.print("Applying blur\n", .{});
    const BlurPostOp = struct {
        divisor: u16,

        pub fn toPixel(self: @This(), in: @TypeOf(blur_kernel)) u8 {
            return @intCast(in.sum() / self.divisor);
        }
    };

    const blurry = try image.applyKernelToImage(
        u8,
        arena.allocator(),
        blur_kernel,
        BlurPostOp{ .divisor = blur_kernel.sum() },
    );

    try writePpmu8(blurry, "blurry.ppm");

    // FIXME: Fold into edge data
    const gradients = try Image(i32).init(arena.allocator(), image.width, image.calcHeight());
    @memset(gradients.data, 0);

    // Output debug image from edges / RLE info
    const rle_back_and_forth = try Image(u8).init(arena.allocator(), image.width, image.calcHeight());

    const debug = try Image(DebugPurpose).init(arena.allocator(), image.width, image.calcHeight());
    @memset(debug.data, .none);

    const edges = try edgePass1(arena.allocator(), blurry);
    const edges2 = try Image(u32).init(arena.allocator(), image.width, image.calcHeight());
    @memset(edges2.data, 0);

    var it = edges2.iter();
    while (it.nextRow()) {
        var row_rle = RleGenerator.init(arena.allocator());

        while (it.nextCol()) {
            row_rle.pushPixel(image.pixel(it.x, it.y).*);

            const gradient = edges.pixel(it.x, it.y).*;
            if (gradient != 0) {
                it.pixel().* = @abs(gradient);
                const edge_dir: RleGenerator.EdgeDir = if (gradient < 0) .light_to_dark else .dark_to_light;
                try row_rle.markEdge(edge_dir);
                continue;
            }

            if (hasHorizontalSibling(edges, it.x, it.y)) continue;

            if (getNearbyGradient(edges, it.x, it.y)) |nearby_gradient| {
                it.pixel().* = @abs(nearby_gradient);
                const edge_dir: RleGenerator.EdgeDir = if (nearby_gradient < 0) .light_to_dark else .dark_to_light;
                try row_rle.markEdge(edge_dir);
            }
        }

        // FIXME: Abusing info that outside should always be light
        try row_rle.markEdge(.light_to_dark);

        var x: usize = 0;
        var length_sum: usize = 0;
        for (row_rle.row.items) |item| {
            length_sum += item.length;
            for (0..item.length) |_| {
                rle_back_and_forth.data[it.y * rle_back_and_forth.width + x] = item.val;
                x += 1;
            }
        }

        const row_start = it.y * debug.width;
        const row_end = (it.y + 1) * debug.width;
        var bcs = BarcodeScanner2{
            .debug = debug.data[row_start..row_end],
            .row = row_rle.row.items,
        };
        var scan_it = bcs.findStartSequence() orelse continue;
        while (scan_it < bcs.row.len) {
            bcs.consumeDark(&scan_it);
            bcs.consumeLight(&scan_it);
        }
        std.debug.assert(x == rle_back_and_forth.width);
    }

    try writeEdgePass1(image, edges, "edges.ppm");
    try writeEdges(image, edges2, "edges2.ppm");
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
