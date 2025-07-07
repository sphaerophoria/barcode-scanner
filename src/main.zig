const std = @import("std");
const stbi = @cImport({
    @cInclude("stb_image.h");
});

const Image = struct {
    data: []u8,
    width: usize,
    height: usize,


    fn load() !Image {
        var widthi: c_int = undefined;
        var heighti: c_int = undefined;
        // FIXME: Leakies :)
        const data = stbi.stbi_load("untitled3.png", &widthi, &heighti, null, 1);

        if (data == null) return error.LoadFailed;

        const width: usize = @intCast(widthi);
        const height: usize = @intCast(heighti);

        return .{
            .data = data[0..width * height],
            .width = width,
            .height = height,
        };
    }

    fn dupe(self: Image, alloc: std.mem.Allocator) !Image {
        return .{
            .data = try alloc.dupe(u8, self.data),
            .width = self.width,
            .height = self.height,
        };
    }
};

const BarcodePreprocessor = struct {
    threshold: u16 = 70,
    in: Image,
    out: Image,

    fn init(alloc: std.mem.Allocator, image: Image) !BarcodePreprocessor {
        return .{
            .in = image,
            .out = try image.dupe(alloc),
        };

    }

    fn process(self: BarcodePreprocessor) !void {

        for (0..self.in.height) |y| {
            for (0..self.in.width) |x| {
                self.processPixelWBlur(x, y);
            }
        }
    }

    fn processPixelWBlur(self: BarcodePreprocessor, x: usize, y: usize) void {
        if (x < 2 or x > self.in.width - 2) return;
        if (y < 2 or y > self.in.height - 2) return;

        var total: u16 = 0;
        for (y - 2..y+2) |iy| {
            for (x-2..x+2) |ix| {
                total += self.in.data[iy * self.in.width + ix];
            }
        }

        const out_val: u8 = if (total > self.threshold * 25) 255 else 0;
        self.out.data[y * self.in.width + x] = out_val;
    }

    fn processPixel(self: BarcodePreprocessor, x: usize, y: usize) void {
        const in_val = self.in.data[y * self.in.width + x];
        const out_val: u8 = if (in_val > self.threshold) 255 else 0;
        self.out.data[y * self.in.width + x] = out_val;
    }
};

const BarcodeScanner = struct {
    image: Image,
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

    fn parseLeft(self: BarcodeScanner, y: usize) ?HalfUpcCode {
        var x, const module_width_u = (self.findStart(y) orelse return null);
        const module_width: i32 = @intCast(module_width_u);
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

    fn readNumberLeft(self: BarcodeScanner, x: *usize, y: usize, module_width: i32) ?u8 {
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

    fn findStart(self: BarcodeScanner, y: usize) ?struct{usize, usize} {
        for (0..self.image.width) |x| {
            const module_width = self.isStartEndSequence(x, y) orelse continue;
            const expected_end_start = ((upca_width_modules - start_end_width_modules) * module_width) + x;

            var offs: i32 = -5;
            while (offs < 5) {
                defer offs += 1;

                if (self.isStartEndSequence(@intCast(@as(i32, @intCast(expected_end_start)) + offs), y)) |_| {
                    //std.debug.print("Found start with width {d} at {d}\n" ,.{module_width, x});
                    return .{x + module_width * 4, module_width};
                }
            }
        }

        return null;
    }

    // return module width if sequence is found
    fn isStartEndSequence(self: BarcodeScanner, x: usize, y: usize) ?usize {
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
            return @intCast(a);
        }

        return null;
    }

    fn countDark(self: BarcodeScanner, x: *usize, y: usize) i32 {
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
    fn countLight(self: BarcodeScanner, x: *usize, y: usize) i32 {
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

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const image = try Image.load();

    const processor = try BarcodePreprocessor.init(arena.allocator(), image);
    try processor.process();

    var ppm = try std.fs.cwd().createFile("out.ppm", .{});
    defer ppm.close();

    var ppm_writer = ppm.writer();

    const scanner = BarcodeScanner {
        .image = processor.out,
    };

    try ppm_writer.print(
        \\P6
        \\{d} {d}
        \\255
        \\
    , .{processor.out.width, processor.out.height});


    for (0..image.height) |y| {
        const start = scanner.findStart(y);
        _ = scanner.parseLeft(y);
        for (0..image.width) |x| {
            if (start != null and start.?[0] == x) {
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(0);
                try ppm_writer.writeByte(255);
            //}
            //else if (scanner.isStartEndSequence(x, y)) |w| {
            //    _ = w;
            //    try ppm_writer.writeByte(255);
            //    try ppm_writer.writeByte(255);
            //    try ppm_writer.writeByte(0);
            } else {
                try ppm_writer.writeByte(processor.out.data[y * image.width + x]);
                try ppm_writer.writeByte(processor.out.data[y * image.width + x]);
                try ppm_writer.writeByte(processor.out.data[y * image.width + x]);
            }
        }
    }
}
