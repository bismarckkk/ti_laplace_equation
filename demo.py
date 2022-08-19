import taichi as ti
import math
ti.init()

screen = (30, 20)
arrowField = (15, 10)
meshSpace = 20
maxItems = 20
refillFrame = 20
refillVelThresh = .1
gui = ti.GUI('demo', tuple(it*meshSpace for it in screen))
guiHeight = meshSpace * screen[1]
vec2 = ti.types.vector(2, float)
V = vec2(1., 0)
dt = .002
fade = 10

points = ti.Vector.field(2, float, screen[0] * screen[1] * 2)
boundaryVel = ti.Vector.field(2, float, (4, max(*screen)))
sources = ti.Struct.field({
    "pos": vec2,
    "q": ti.f32
}, shape=maxItems)
vortexes = ti.Struct.field({
    "pos": vec2,
    "q": ti.f32
}, shape=maxItems)
dipoles = ti.Struct.field({
    "pos": vec2,
    "m": ti.f32
}, shape=maxItems)
arrows = ti.Struct.field({
    "dir": vec2,
    "vel": ti.f32
}, shape=arrowField)
start = ti.field(ti.i32, shape=2)
lastStart = ti.field(ti.i32, shape=2)
points.fill(-100)


@ti.kernel
def initPoints():
    # for x, y in ti.ndrange(*screen):
    #     points[x + y * screen[0]][0] = (x + 0.5) / screen[0]
    #     points[x + y * screen[0]][1] = (y + 0.5) / screen[1]
    dipoles[0].pos = vec2(0.5, 0.5)
    dipoles[0].m = 0.01
    vortexes[0].pos = vec2(0.5, 0.5)
    vortexes[0].q = -0.5


@ti.func
def getVel(pos):
    vel = V
    for i in range(maxItems):
        uv = pos - sources[i].pos
        uv[0] *= screen[1] / screen[0]
        vel += uv * sources[i].q / (2 * ti.math.pi * (uv[0] ** 2 + uv[1] ** 2))

        uv = pos - vortexes[i].pos
        uv = vec2(-uv[1], uv[0])
        uv[0] *= screen[1] / screen[0]
        vel += uv * vortexes[i].q / (2 * ti.math.pi * (uv[0] ** 2 + uv[1] ** 2))

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
def updateBoundaryVel():
    for i, j in ti.ndrange(4, ti.max(*screen)):
        if i == 0:
            if j >= screen[1]:
                continue
            pos = vec2(0, (j + .5) / screen[1])
            boundaryVel[i, j] = getVel(pos)
        elif i == 1:
            if j >= screen[0]:
                continue
            pos = vec2((j + .5) / screen[0], 0)
            boundaryVel[i, j] = getVel(pos)
        elif i == 2:
            if j >= screen[1]:
                continue
            pos = vec2(1, (j + .5) / screen[1])
            boundaryVel[i, j] = getVel(pos)
        elif i == 3:
            if j >= screen[0]:
                continue
            pos = vec2((j + .5) / screen[0], 1)
            boundaryVel[i, j] = getVel(pos)


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
                    if boundaryVel[0, start[1]][0] > refillVelThresh:
                        points[i] = vec2(0, (start[1] + .5) / screen[1])
                        break
                elif start[0] == 1:
                    if boundaryVel[1, start[1]][1] > refillVelThresh:
                        points[i] = vec2((start[1] + .5) / screen[0], 0)
                        break
                elif start[0] == 2:
                    if boundaryVel[2, start[1]][0] < -refillVelThresh:
                        points[i] = vec2(1, (start[1] + .5) / screen[1])
                        break
                elif start[0] == 3:
                    if boundaryVel[3, start[1]][1] < -refillVelThresh:
                        points[i] = vec2((start[1] + .5) / screen[0], 1)
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
        for j in range(maxItems):
            if sources[j].q < 0 and ti.math.length(points[i] - sources[j].pos) < 0.025:
                points[i] = vec2(-100, -100)
            if dipoles[j].m != 0 and ti.math.length(points[i] - dipoles[j].pos) < 0.05:
                points[i] = vec2(-100, -100)


@ti.kernel
def updateArrows():
    for x, y in ti.ndrange(*arrowField):
        pos = vec2((x + 1) * (1 - 1 / arrowField[0]) / arrowField[0], (y + 1) * (1 - 1 / arrowField[1]) / arrowField[1])
        vel = getVel(pos)
        arrows[x, y].vel = vel.norm()
        arrows[x, y].dir = vel / vel.norm() / meshSpace / 1.5


def drawArrows(gui):
    global fade
    if fade < 10:
        fade += 1
    if fade == 0:
        updateArrows()
    arr = arrows.to_numpy()
    vel = arr['vel'].reshape(1, -1)[0]
    vel = ((vel / vel.max() * 0x88 + 0x55) * (math.fabs(fade / 10))).astype(int)
    vel *= 2 ** 16 + 2 ** 8 + 1
    gui.arrow_field(arr['dir'], radius=1.5, color=vel, bound=1)


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
    for i in range(maxItems):
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
    global fade
    while gui.get_event((ti.GUI.PRESS, ti.GUI.LMB), (ti.GUI.PRESS, ti.GUI.RMB)):
        if gui.is_pressed(ti.GUI.LMB) and gui.is_pressed(ti.GUI.RMB):
            for i in range(maxItems):
                if sources[i].q != 0 and (sources[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    sources[i].q = 0
                if vortexes[i].q != 0 and (vortexes[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    vortexes[i].q = 0
                if dipoles[i].m != 0 and (dipoles[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                    dipoles[i].m = 0
        else:
            if gui.is_pressed('s'):
                for i in range(maxItems):
                    if sources[i].q == 0:
                        if gui.is_pressed(ti.GUI.RMB):
                            sources[i].q -= 1
                        else:
                            sources[i].q += 1
                        sources[i].pos = vec2(*gui.get_cursor_pos())
                        break
            elif gui.is_pressed('v'):
                for i in range(maxItems):
                    if vortexes[i].q == 0:
                        if gui.is_pressed(ti.GUI.RMB):
                            vortexes[i].q -= 0.5
                        else:
                            vortexes[i].q += 0.5
                        vortexes[i].pos = vec2(*gui.get_cursor_pos())
                        break
            elif gui.is_pressed('d'):
                for i in range(maxItems):
                    if dipoles[i].m == 0:
                        if gui.is_pressed(ti.GUI.RMB):
                            dipoles[i].m -= 0.01
                        else:
                            dipoles[i].m += 0.01
                        dipoles[i].pos = vec2(*gui.get_cursor_pos())
                        break
            else:
                for i in range(maxItems):
                    if sources[i].q != 0 and (sources[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                        if gui.is_pressed(ti.GUI.RMB):
                            sources[i].q -= 0.5 * int(sources[i].q >= 0.0) - (sources[i].q <= 0.0)
                        else:
                            sources[i].q += 0.5 * int(sources[i].q >= 0.0) - (sources[i].q <= 0.0)
                    if vortexes[i].q != 0 and (vortexes[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                        if gui.is_pressed(ti.GUI.RMB):
                            vortexes[i].q -= 0.1 * int(vortexes[i].q >= 0.0) - (vortexes[i].q <= 0.0)
                        else:
                            vortexes[i].q += 0.1 * int(vortexes[i].q >= 0.0) - (vortexes[i].q <= 0.0)
                    if dipoles[i].m != 0 and (dipoles[i].pos - vec2(*gui.get_cursor_pos())).norm() < 5 / guiHeight:
                        if gui.is_pressed(ti.GUI.RMB):
                            dipoles[i].m -= 0.001 * int(dipoles[i].m >= 0.0) - (dipoles[i].m <= 0.0)
                        else:
                            dipoles[i].m += 0.001 * int(dipoles[i].m >= 0.0) - (dipoles[i].m <= 0.0)
        fade = -math.fabs(fade)


if __name__ == '__main__':
    initPoints()
    updateBoundaryVel()
    updateArrows()
    refillPoints()
    refillCount = 0
    frame = 0
    while gui.running:
        processGuiEvent(gui)
        drawArrows(gui)
        updatePoints()
        ps = points.to_numpy()
        gui.circles(ps, 3, color=0x2E94B9)
        if refillCount > refillFrame:
            refillCount = 0
            updateBoundaryVel()
            refillPoints()
        drawMark(gui, frame)
        # gui.show(f'./frames/frame_{frame}.png')
        gui.show()
        refillCount += 1
        frame += 1
