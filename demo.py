import taichi as ti
import math
ti.init(arch=ti.cpu, cpu_max_num_threads=1)

screen = (30, 20)
meshSpace = 20
gui = ti.GUI('demo', tuple(it*meshSpace for it in screen))
guiHeight = meshSpace * screen[1]
vec2 = ti.types.vector(2, float)
V = vec2(1., 0)
dt = .002

points = ti.Vector.field(2, float, screen[0] * screen[1] * 2)
sources = ti.Struct.field({
    "pos": vec2,
    "q": ti.f32
}, shape=20)
vortexes = ti.Struct.field({
    "pos": vec2,
    "q": ti.f32
}, shape=20)
dipoles = ti.Struct.field({
    "pos": vec2,
    "m": ti.f32
}, shape=20)
start = ti.field(ti.i32, shape=2)
lastStart = ti.field(ti.i32, shape=2)
points.fill(-100)


@ti.kernel
def initPoints():
    for x, y in ti.ndrange(*screen):
        points[x + y * screen[0]][0] = (x + 0.5) / screen[0]
        points[x + y * screen[0]][1] = (y + 0.5) / screen[1]
    dipoles[0].pos = vec2(0.5, 0.5)
    dipoles[0].m = 0.01
    vortexes[0].pos = vec2(0.5, 0.5)
    vortexes[0].q = -0.5


@ti.func
def getVel(pos):
    vel = V
    for i in range(20):
        uv = pos - sources[i].pos
        uv[0] *= screen[1] / screen[0]
        vel += uv * sources[i].q / (2 * ti.math.pi * (uv[0] ** 2 + uv[1] ** 2))
    for i in range(20):
        uv = pos - vortexes[i].pos
        uv = vec2(-uv[1], uv[0])
        uv[0] *= screen[1] / screen[0]
        vel += uv * vortexes[i].q / (2 * ti.math.pi * (uv[0] ** 2 + uv[1] ** 2))
    for i in range(20):
        uv = pos - dipoles[i].pos
        uv[0] *= screen[1] / screen[0]
        vel_t = vec2(uv[1]**2 - uv[0]**2, -2*uv[0]*uv[1])
        vel += vel_t * dipoles[i].m / (uv[0] ** 2 + uv[1] ** 2) ** 2
    return vel


@ti.func
def updateStart():
    if start[0] == 0:
        start[1] += 1
        if start[1] >= screen[1]:
            start[0] = 1
            start[1] = 0
    elif start[0] == 2:
        start[1] += 1
        if start[1] >= screen[1]:
            start[0] = 0
            start[1] = 0
    elif start[0] == 1:
        start[0] = 3
    else:
        start[1] += 1
        start[0] = 1
        if start[1] >= screen[0]:
            start[0] = 2
            start[1] = 0


@ti.kernel
def refillPoints():
    ti.loop_config(serialize=True)
    started = False
    for i in range(points.shape[0]):
        if points[i][0] == -100 and points[i][1] == -100:
            while start[0] != lastStart[0] or start[1] != lastStart[1]:
                if started:
                    updateStart()
                else:
                    started = True
                if start[0] == 0:
                    pos = vec2(0, (start[1] + .5) / screen[1])
                    vel = getVel(pos)
                    if vel[0] > .1:
                        points[i] = pos
                        break
                elif start[0] == 1:
                    pos = vec2((start[1] + .5) / screen[0], 0)
                    vel = getVel(pos)
                    if vel[1] > .1:
                        points[i] = pos
                        break
                elif start[0] == 2:
                    pos = vec2(1, (start[1] + .5) / screen[1])
                    vel = getVel(pos)
                    if vel[0] < -.1:
                        points[i] = pos
                        break
                elif start[0] == 3:
                    pos = vec2((start[1] + .5) / screen[0], 1)
                    vel = getVel(pos)
                    if vel[1] < -.1:
                        points[i] = pos
                        break
    lastStart[0] = start[0]
    lastStart[1] = start[1]
    updateStart()


@ti.kernel
def updatePoints():
    for i in points:
        if 0 <= points[i][0] <= 1 and 0 <= points[i][1] <= 1:
            vel = getVel(points[i])
            points[i] += vel * dt
        else:
            points[i] = vec2(-100, -100)
        for j in range(20):
            if sources[j].q < 0 and ti.math.length(points[i] - sources[j].pos) < 0.025:
                points[i] = vec2(-100, -100)
            if ti.math.length(points[i] - dipoles[j].pos) < 0.05:
                points[i] = vec2(-100, -100)


def drawMark(gui, frame):
    triangleTrans = [
        vec2(0, 1) / (guiHeight),
        vec2(math.cos(7./6. * math.pi), math.sin(7./6. * math.pi)) / (guiHeight),
        vec2(math.cos(-1./6. * math.pi), math.sin(-1./6. * math.pi)) / (guiHeight)
    ]
    rectTrans = [
        vec2(1 * screen[1] / screen[0], 1) / (guiHeight),
        vec2(-1 * screen[1] / screen[0], -1) / (guiHeight),
    ]
    for i in range(20):
        if dipoles[i].m > 0:
            gui.circle(dipoles[i].pos, 0xFFFDC0, dipoles[i].m * 2000)
        elif dipoles[i].m < 0:
            gui.circle(dipoles[i].pos, 0xD25565, dipoles[i].m * -2000)
        if sources[i].q > 0:
            gui.rect(
                sources[i].pos + rectTrans[0] * 2 * sources[i].q,
                sources[i].pos + rectTrans[1] * 2 * sources[i].q,
                2 * sources[i].q + 1, 0xFFFDC0
            )
        elif sources[i].q < 0:
            gui.rect(
                sources[i].pos + rectTrans[0] * 2 * sources[i].q,
                sources[i].pos + rectTrans[1] * 2 * sources[i].q,
                -2 * sources[i].q + 1, 0xD25565
            )
        if vortexes[i].q != 0:
            rotateMatrix = ti.Matrix([
                [math.cos(vortexes[i].q * dt * 40 * frame), -math.sin(vortexes[i].q * dt * 40 * frame)],
                [math.sin(vortexes[i].q * dt * 40 * frame), math.cos(vortexes[i].q * dt * 40 * frame)]
            ])
            trans = [rotateMatrix @ it for it in triangleTrans]
            for it in trans:
                it[0] *= screen[1] / screen[0]
            gui.triangle(
                vortexes[i].pos + trans[0] * 16,
                vortexes[i].pos + trans[1] * 16,
                vortexes[i].pos + trans[2] * 16,
                0xD25565
            )


def processGuiEvent(gui):
    if gui.get_event((ti.GUI.PRESS, ti.GUI.LMB)):
        if gui.is_pressed('s'):
            for i in range(20):
                if sources[i].q == 0:
                    if gui.is_pressed(ti.GUI.CTRL):
                        sources[i].q -= 1
                    else:
                        sources[i].q += 1
                    sources[i].pos = vec2(*gui.get_cursor_pos())
                    break
        elif gui.is_pressed('v'):
            for i in range(20):
                if vortexes[i].q == 0:
                    if gui.is_pressed(ti.GUI.CTRL):
                        vortexes[i].q -= 0.5
                    else:
                        vortexes[i].q += 0.5
                    vortexes[i].pos = vec2(*gui.get_cursor_pos())
                    break
        elif gui.is_pressed('d'):
            for i in range(20):
                if dipoles[i].m == 0:
                    if gui.is_pressed(ti.GUI.CTRL):
                        dipoles[i].m -= 0.01
                    else:
                        dipoles[i].m += 0.01
                    dipoles[i].pos = vec2(*gui.get_cursor_pos())
                    break
        elif gui.is_pressed('r'):
            for i in range(20):
                if sources[i].q != 0 and (sources[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    sources[i].q = 0
                if vortexes[i].q != 0 and (vortexes[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    vortexes[i].q = 0
                if dipoles[i].m != 0 and (dipoles[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    dipoles[i].m = 0
        else:
            for i in range(20):
                if sources[i].q != 0 and (sources[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    if gui.is_pressed(ti.GUI.CTRL):
                        sources[i].q -= 0.5 * int(sources[i].q >= 0.0) - (sources[i].q <= 0.0)
                    else:
                        sources[i].q += 0.5 * int(sources[i].q >= 0.0) - (sources[i].q <= 0.0)
                if vortexes[i].q != 0 and (vortexes[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    if gui.is_pressed(ti.GUI.CTRL):
                        vortexes[i].q -= 0.1 * int(vortexes[i].q >= 0.0) - (vortexes[i].q <= 0.0)
                    else:
                        vortexes[i].q += 0.1 * int(vortexes[i].q >= 0.0) - (vortexes[i].q <= 0.0)
                if dipoles[i].m != 0 and (dipoles[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    if gui.is_pressed(ti.GUI.CTRL):
                        dipoles[i].m -= 0.001 * int(dipoles[i].m >= 0.0) - (dipoles[i].m <= 0.0)
                    else:
                        dipoles[i].m += 0.001 * int(dipoles[i].m >= 0.0) - (dipoles[i].m <= 0.0)


if __name__ == '__main__':
    initPoints()
    refillPoints()
    refillCount = 0
    frame = 0
    while gui.running:
        processGuiEvent(gui)
        updatePoints()
        ps = points.to_numpy()
        gui.circles(ps, 3, color=0x2E94B9)
        if refillCount > 20:
            refillCount = 0
            refillPoints()
        drawMark(gui, frame)
        gui.show(f'./frames/frame_{frame}.png')
        refillCount += 1
        frame += 1
