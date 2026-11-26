# collision_proto — 3DGS 碰撞体原型（TSDF 路线）

把一个**显式 3DGS PLY** 变成可以给引擎用的碰撞网格。不需要训练相机：

```
PLY ──► 虚拟航拍相机渲染深度 ──► TSDF 融合 ──► surface nets 网格 ──► 清理 / 减面
        (俯视 + 两层斜视，                                    ├► collision_lite.glb   碰撞代理
         切平面中值深度)                                       ├► ground_net.glb       地面兜底网
                                                             └► tsdf.npz             SDF 碰撞用
```

所有输出都在 **splat 的原始坐标系**里，和 PLY 直接叠在一起，不需要任何变换。

## 用法

```bat
:: 第一次（或换了终端环境后）要先编译 gsplat 的 CUDA 后端
py.bat build_gsplat.py

:: 俯视图：确认朝上方向、量尺度、挑区块（网格线坐标是场景单位）
py.bat overview.py D:/3DGS_Data/with_aerial.ply --out out/overview

:: 生成碰撞体（--center 用场景单位，其余长度都是米；--mpu = 每场景单位多少米）
py.bat splat_to_collision.py D:/3DGS_Data/with_aerial.ply --out out/chunk_a --mpu 58 --center -0.55 -0.1 --size 100 100

:: 检查：哪里能站、连成多大一片、哪里会漏下去（--net 1 = 网格 + 兜底网一起算）
py.bat walkability.py out/chunk_a
py.bat walkability.py out/chunk_a --net 1

:: 同一机位下 splat | 碰撞网格线框 | 叠加。每 6 个数一组：x y(场景单位) 离地高度(m) 偏航 俯角 FOV
py.bat preview_mesh.py out/chunk_a --views -1.00 -0.60 20 45 30 70
```

`py.bat` 必须用：它负责 MSVC 环境、代码页和 venv。原因见 `build_gsplat.py` 顶部注释和 `gs_core._msvc_jit_shim`。
跑的过程中不要改 `py.bat`（cmd 是边读边执行批处理的）。

## 文件

| 文件 | 作用 |
|---|---|
| `gs_core.py` | 读 PLY、估计水平坐标系、gsplat 渲染、**切平面中值深度** |
| `tsdf_core.py` | 稠密 TSDF（torch）、surface nets、SDF 光线步进、连通域清理、顶点聚类减面 |
| `splat_to_collision.py` | 主流程 |
| `walkability.py` | 胶囊体尺度的可站立 / 连通 / 漏洞检查 |
| `preview_mesh.py` | 无依赖的网格光栅化预览 |
| `overview.py` | 俯视总览 |
| `xsection.py` | 融合体积的竖直剖面 |
| `depth_consistency.py` | 跨视角深度一致性（中心深度 vs 切平面深度）；目标点是针对 `with_aerial.ply` 写死的 |
| `test_synthetic.py` | 解析场景（地面 + 盒子）上的自测，不需要 gsplat |
| `voxel_collision.py` | 体素化路线：密度或视角雕刻 → 可站立 → 可走区域 → 空气或步行雕刻 → 外壳碰撞网格 |
| `voxel_mesh.py` | 二值体素的面合并网格化、开闭运算去噪；直接运行是合成房间自测 |
| `camera_volume.py` | 相机体积的采样、自由相机夹回、第三人称弹簧臂（numpy + scipy，不依赖引擎和 GPU）；直接运行是带渲染的检查 |
| `ground_patch.py` | 地面补丁（贴地高度场，补洞）、物体网格（碰撞网格去掉地面）、抬高的兜底网 v2 |
| `export_ue.py` | 转到 UE 坐标（水平、区块中心为原点、厘米）导出 GLB，顶点色取自 splat |
| `ue/setup_level.py` | 在 UE 里导入网格、建试走关卡（UE 端 Python） |
| `ue/walk_test.py` | PIE 自动试走：64 个第三人称角色随机行走，记录掉落 / 卡住 / 被困（UE 端 Python） |
| `ue/plot_walk.py`、`ue/diag_walk.py` | 把试走轨迹和事件画到可站立图 / 补丁地形上 |
| `ue/camera_eval.py` | 试走里记录的弹簧臂相机：拿相机体积检查，渲染相机看到的画面 |
| `ue/probe_import.py` | 实测某个 UE 版本的 glTF 导入轴向 |

## 深度是怎么来的（这是质量的关键）

gsplat 只给"期望深度"，而且每个高斯用的是**中心**的深度。直接拿来做 TSDF 有两个问题，都在这里处理了：

1. **浮游物和背景把深度拉偏。** 每个视角按深度切 96 层各渲一次（远平面裁剪），得到不透明度的精确前缀和，
   取"累积不透明度过总量一半"那一层内的加权平均深度。切片范围只覆盖区块体积，所以同样层数下切片更薄。
2. **大而斜的高斯变成朝向相机的平板。** 射线上高斯密度最大处的深度是
   `t*(r) = (rᵀPμ)/(rᵀPr)`，在中心射线附近一阶展开后对像素坐标是仿射的。把三个仿射系数当作颜色通道渲染，
   就得到按切平面逐像素计算的合成深度，不需要改光栅化器。

在 `with_aerial.ply` 上，跨视角深度中位差（越小越适合融合）：

| 深度 | 40 层 | 96 层 | 200 层 |
|---|---|---|---|
| 高斯中心 | 0.42 m | 0.37 m | 0.35 m |
| 切平面 | 0.32 m | 0.23 m | 0.19 m |

（`depth_consistency.py`，4 个目标点各 7 个视角两两互投；切片范围固定为 8–150 m，比主流程里的更粗。）

## 融合与网格

- **空射线也参与融合**：总不透明度 < 0.05 的射线把沿途体素标成"已观测、为空"。
- **自由空间证据降权**（`--free-weight 0.3`）：只被一部分视角看到的地面不会被其余视角否决掉，代价是浮游物去除变弱。
- **悬空碎块剔除**：最低点高出地形 3 m 以上的连通块删掉；面积 < 5 m² 的碎片删掉。
- **地面兜底网** `ground_net.glb`：12 m 窗口的低分位地形下沉 0.25 m，接住从地面破洞掉下去的角色；大屋顶下方会被裁掉。
- `--smooth`（融合后高斯平滑）默认关闭：在测试场景上它去掉的地板比补上的多。
- `--street 1`（街景相机补扫）默认关闭且未经验证：测试用的 splat 在 1.6 m 视高渲染出来全是拉长的大高斯。

## 在 `with_aerial.ply` 一个 100 m × 100 m 区块上的结果（`out/chunk_a`）

- 809 个虚拟视角，RTX 5070 Ti 上整条管线 78 秒。
- `collision_full.ply` 304 万三角形（52 MB），`collision_lite.glb` 26 万三角形（4.5 MB）。
- 7 个留出视角里有 6 个：融合表面与 splat 深度的中位差 6–10 cm，94–99.6% 的 splat 表面像素能找到融合表面。
  第 7 个在堆场（脚手架、材料堆）上方：中位差 0.37 m，89.5%。
- 可站立面积占区块 96%。仅网格：地面连成 4288 m²，屋顶 3295 m²，地面连通区有 17% 的格子挨着洞或落差。
  网格 + 兜底网：地面连通区 5644 m²，挨着落差的格子降到 4.3%。
  **这些数字偏乐观**：`walkability.py` 只要 0.3 m 格子里任意一列 15 cm 体素有地板就算可站，把亚米级的洞藏掉了。
  实际上地面连通区里只有 75% 的格子在网格上真有地板（格子中心 0.5 m 内），角色会掉下去。见下面 UE 试走。

## 导入 UE 试走（UE 5.8）

SceneFarm 是 UE 5.5 工程，这台机器只有 UE 5.8，所以用的是单独的测试工程
`D:\3DGS_Research\collision_ue\CollisionWalk.uproject`（第三人称模板），每个运行目录一个关卡：`/Game/Splat/<运行名>/Lvl_<运行名>`（例如 `/Game/Splat/chunk_a/Lvl_chunk_a`、`/Game/Splat/vox_carve/Lvl_vox_carve`）。

```bat
py.bat walkability.py out/chunk_a          :: 同时写出 walk_cells.npz
py.bat ground_patch.py out/chunk_a         :: ground_patch.ply、collision_objects.ply、ground_net2.ply
py.bat export_ue.py out/chunk_a            :: out/chunk_a/ue/*.glb + scene_ue.json
```
```powershell
$ue = "C:\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor.exe"
& $ue D:\3DGS_Research\collision_ue\CollisionWalk.uproject -ExecutePythonScript="D:/3DGS_Research/collision_proto/ue/setup_level.py D:/3DGS_Research/collision_proto/out/chunk_a/ue" -unattended
& $ue D:\3DGS_Research\collision_ue\CollisionWalk.uproject -ExecutePythonScript="D:/3DGS_Research/collision_proto/ue/walk_test.py D:/3DGS_Research/collision_proto/out/chunk_a/ue" -unattended
```
然后 `py.bat ue/plot_walk.py out/chunk_a`、`py.bat ue/diag_walk.py out/chunk_a`。两个编辑器命令跑完自动退出；
自己走就打开工程、加载对应关卡按 Play。

关卡里的碰撞（场景里没有 3DGS 渲染插件，彩色网格只是 splat 的替身，本身不参与碰撞）：

| 演员 | 作用 | 游戏中可见 |
|---|---|---|
| `SplatCollision` | 带 splat 顶点色的 lite 网格，只用来看 | 是 |
| `GroundPatch` | 地面补丁：地面连通区的高度场（亚体素高度、5×5 中值），补 5 m 内的洞，身体高度有东西的地方不补 | 否 |
| `CollisionObjects` | lite 网格去掉补丁上 0.3 m 以内及以下的三角形：墙、车、货堆 | 否 |
| `GroundNet` | 兜底网 v2：地面 10 m 内贴在外推地面下 0.25 m，更远处用原来的 DTM 网 | 否 |
| `Wall_*` | 区块边界的隐形墙 | 否 |

### 试走结果

64 个模板角色（胶囊半径 35 cm、半高 90 cm、台阶 45 cm、可走坡度 44.8°、600 cm/s）由 AI 控制器随机行走，
跑的是真实的 CharacterMovement 和 Chaos 碰撞，每轮 45 s，出生点在地面连通区。
"行走效率" = 实际行走距离 ÷ 满速一直走的距离。"被困" = 4 m 内 12 s 里卡住 3 次，之后重新出生。

| 碰撞配置 | 行走效率 | 掉出世界 | 被困 | 被挡住 / 无遮挡卡住 |
|---|---|---|---|---|
| 仅网格 | 0.17 | 333 | 47 | 134 / 158 |
| 网格 + 补丁 | 0.37 | 54 | 69 | 348 / 153 |
| 网格 + 补丁 + 兜底网 | 0.38 | 0 | 64 | 465 / 103 |
| 补丁 + 物体 + 兜底网（关卡默认） | 0.39 | 1 | 70 | 465 / 105 |

同样设置多跑几次，各项会差 10–20%（帧时间不确定）。

- 位置、尺度都对：导入后的包围盒和导出值差 0.0 cm；角色走在地面连通区里，上不了屋顶；没有角色越过边界墙。
- 只用融合网格不能走：每 45 s 掉出世界 300 多次。
- 补丁是主要改进；兜底网 v2 把掉出世界压到 0–1。原来的 DTM 兜底网在地面附近平均比实际地面低 1.7 m，
  掉上去就是坑；v2 只低 0.25 m，网上的被困从 57 次降到 26–40 次。
- 剩下的被困和卡住大多在杂物区：卡车和挂车（补丁把车顶接进地面，四周是陡边）、堆场里一格格的高差、
  车底和货物附近没补的洞（`ue/diag_split.png`）。这些地方重建本身就乱，地面处理解决不了。

### UE 5.8 的坑

- Interchange 的 glTF 导入把 glTF (x, y, z) 米映射成 UE (x, z, y) 厘米（`ue/probe_import.py` 实测）。
- Interchange 默认开 Nanite，复杂碰撞会用简化过的回退网格（26 万 → 5 万三角形），`setup_level.py` 里关掉了。
- `-ExecutePythonScript` 在脚本返回后就关编辑器，要先 `unreal.EditorPythonScripting.set_keep_python_script_alive(True)`。
- `-run=pythonscript` 命令行模式不能往关卡里放 actor；Python 也不能往 PIE 世界里生成 actor，
  所以角色是开 PIE 前摆在编辑器关卡里（不保存）。
- 工程配置里关掉了 `bThrottleCPUWhenNotForeground`，否则编辑器在后台时 PIE 会降帧。
- 编辑器退出时偶尔报访问冲突，发生在日志关闭之后，资产已保存。
- PIE 里用 `set_editor_property` 改蓝图组件的属性，默认会发属性变更通知，触发构造脚本重跑，组件被换成新的，手里的引用就失效了。要传 `unreal.PropertyAccessChangeNotifyMode.NEVER`。
- AI 控制器会把控制旋转的俯仰强制成 0，所以弹簧臂不能用 Pawn 控制旋转，要设成绝对旋转后直接设世界旋转。

## 体素化路线（`voxel_collision.py`）

一个脚本，室内室外同一套流程，模式只改参数：

```bat
:: 有 TSDF 运行结果（相机视角雕刻）时：推荐
py.bat voxel_collision.py <ply> --out out/vox_air --mpu 58 --mode outdoor --source carve --carve-run out/chunk_a
:: 没有相机、splat 表面比较干净时：直接对高斯做体素化（类似 PlayCanvas splat-transform）
py.bat voxel_collision.py <ply> --out out/vox_a --mpu 58 --center -0.55 -0.1 --size 100 100 --mode outdoor
py.bat export_ue.py out/vox_air            :: 然后 ue/setup_level.py、ue/walk_test.py 同上
py.bat voxel_mesh.py                       :: 面合并自测：合成房间 72 个三角形，体积和体素数一致
```

1. **实心从哪来**
   - `--source density`：每个高斯按不透明度 × 高斯核加到体素里。两个阈值：能站的 ≥ 0.1（真实地表很薄，密度常常只有 0.1–0.5），能挡人的 ≥ 0.5（不让半透明的雾和浮游物把路堵死）。
   - `--source carve`：任何视角看穿过的空间为空，其余（表面后面、地下、看不见的室内、拍摄范围外）一律实心。
2. **可站立**：脚下有实心，胶囊体（半径 0.35 m、高 1.8 m）放得下，最低 0.45 m 当台阶跨过。
3. **可走区域**：室外取所有 ≥ 50 m² 的连通片，因为区域之间常常只能跳下去、走不上来；室内和单物体从种子点泛洪。
4. **补地面**：可走区域 5 m 内没有地板的洞，按周边高度补一层，身体高度有东西的地方不补。
5. **空的部分**，两种（`--free`）：
   - `air`（室外默认，借鉴 PlayCanvas 的雕刻）：先在可走地面以下铺一层"地基"，按可走地面高度外推到整个区块，20 m 内都没有可走地面的列整列实心；然后胶囊体从种子点在空气里三维泛洪，能到的就是空。只能跳下去的区域会经由上方空气连起来，相机也能用。和 PlayCanvas 室外"从下往上填到第一个表面"不同：地基跟着可走地面走，不会停在地下杂物团上。
   - `walk`（室内默认）：可走位置上胶囊体扫过的空间，往上延伸到第一个障碍，再加上障碍物以上的开放空气。
6. **外壳**：空以外全部实心，边界就是碰撞网格。站在上面的角色**掉不进洞、也走不出可走范围，这是构造本身保证的**，不需要补丁和兜底网。
7. **网格**（`--mesh`）：
   - `faces`（默认）：体素面合并成矩形，跟体素完全一致、闭合；之前先做一轮 3×3×3 开闭运算去掉单体素噪声。
   - 体素大小：density 来源用 `--voxel`；carve 来源跟着 TSDF 网格走，用 `--coarsen 2` 把 2×2×2 合成一个（多数为实心才算实心）。台阶高度向下取整成整体素，0.3 m 时是 1 格。
   - `qem`：先平滑成 surface nets 网格，再用 QEM 减面（`fast-simplification`），减到 p99.9 偏差不超过 `--qem-tol`（默认 11 cm）为止。
   - `smooth`：surface nets 加顶点聚类。
   - 显示用的彩色网格：装了 `fast-simplification` 就固定减 95%。
8. **出生点**：只导出落在最终空间里、身体放得下的落脚点。空气雕刻到不了的小空腔在外壳里是实心，其中的落脚点会被去掉（这一块约占 24%）。

### 在 `with_aerial.ply` 同一区块上

| | 用时 | 可走面积 | 碰撞三角形 | UE 试走：效率 / 掉出世界 / 被困 |
|---|---|---|---|---|
| TSDF 网格 + 补丁 + 物体 + 兜底网 | 78 s + 后处理 | 地面约 5400 m² | 26 万 + 15 万 + 11 万 + 16 万 | 0.39 / 0–1 / 约 70 |
| 体素外壳，carve 来源，按步行雕刻，平滑网格 | 7 s（复用 TSDF 体积） | 8963 m² | 29.5 万（一个网格） | 0.33 / **0** / 105 |
| **体素外壳，carve 来源，空气雕刻，面合并（默认）** | 7 s | 8963 m² | 26.9 万（一个网格） | **0.45–0.46** / **0** / **30–33** |
| 同上，体素加粗到 0.3 m（`--coarsen 2`） | 3 s | 9239 m² | 6.7 万 | 0.42 / 0 / 38 |
| 同上，QEM ≤ 11 cm | 20 s | 8963 m² | 5.6 万 | 0.43 / 0 / 46 |
| 同上，QEM ≤ 6 cm | 25 s | 8963 m² | 11.1 万 | 0.40 / 0 / 45 |
| 体素外壳，density 来源 | 13 s | 9702 m² | 28 万 | 没进 UE：厂房里的杂物区也被算成可走 |

- **density 来源在这个航拍 splat 上不行。** 地下和厂房里塞满了被遮挡的大团高斯，密度和真实表面一样高。地表那层又是半透明的，光能透下去，所以这些大团本身就参与了地面的外观，可见性过滤也只删掉 9%。我加了"大屋顶下不可走""比周边低 2 m 以上算陷进去"两条室外规则，只能缓解。
- **carve 来源干净**：没有空腔，也没有悬空地板。厂房屋顶也算可走面，但从地面上不去，所以没有影响。
- **空气雕刻 + 面合并是目前最好的组合**。修正出生点后跑了两次：行走效率 0.45–0.46，被困 30–33 次，无遮挡卡住 13–17 次。按步行雕刻时区域边缘紧贴胶囊体，被困 105 次。修正出生点前的一次是 0.48 / 被困 20，那一轮有一部分出生点落在实心里。剩下的被困集中在区块西边界和一块下沉的方形区域边缘（角色跳进去后在里面能走，但爬不出来）。
- **QEM 网格轻 5 倍，但走起来更容易卡**：无遮挡卡住 38–43 次（面合并 13–17 次）。把容差从 11 cm 收紧到 6 cm 也没有改善，说明问题不在减面误差，而在平滑表面本身：UE 的角色移动遇到 15 cm 一级的竖直踏步能直接跨上去，遇到不规则的斜面和鼓包反而会卡。所以碰撞默认用体素面，QEM 留给三角形预算紧或只给相机用的场合。
- **要更轻，加粗体素比 QEM 好**：0.3 m 体素面只有 6.7 万三角形，无遮挡卡住只有 9 次（QEM 是 38–43 次）。代价是窄处封得更早：被挡住 522 次（0.15 m 是 464–480 次），被困 38 次（30–33 次）；地面没有系统偏差，但会浮起或下陷约一个细体素（15 cm，p90 为 30 cm），近看能察觉。
- **混合网格不值得做**：96% 的三角形在离可走地面 1.8 m 以内，头顶以上全换成 QEM 最多省 4%。
- **无损 QEM 不如矩形合并**：对未合并的体素面只做共面合并，得到 31.3 万三角形，比按行合并的 26.9 万还多。
- **面合并在室外省得不多**：26.9 万，比平滑网格少 9%、比不先去噪的 34.9 万少 23%。室外地形每 15 cm 高差一级台阶，每级一圈小矩形；合成房间只要 72 个三角形，平整的室内才会小一个数量级。

### 局限

- **室内模式没测过**：手头没有室内 splat。`ground_only.ply` 是沿街拍摄的，竖直方向自动估计不对，和航拍版本也不在同一坐标系，没用上。
- **carve 来源现在要借 TSDF 运行的体积。** 有训练相机时，应该直接用训练相机做雕刻，这还没实现。
- **0.15 m 精确网格在室外约 27 万三角形。** 要更轻就用 `--coarsen 2`（0.3 m，6.7 万），接受约 15 cm 的地面误差和更窄的通行能力。
- **墙的高度**：可走范围外的隐形墙一直通到格子顶部。

### 相机体积（`--camera 1`，默认开）

人物碰撞（`collision.ply`）只管角色。相机会飞，近裁面不能进 splat，也不能飞到拍摄质量差的地方，所以单独算一个"相机可以待的空间"：

- **空的部分里**（上面第 5 步的空间）；
- **离任何可见的东西至少 `--cam-clear`（0.3 m）**：可见的东西包括能站的实心、障碍，以及密度 ≥ `--cam-opacity`（0.03）的稀疏植被和雾（去掉小于 `--min-blob` 的碎块）。可达空间自身的边界不算可见物，它本来就离障碍一个胶囊半径；
- **不高于可走地面以上 `--cam-max`**：室外默认 5 m，按每列最高的可走落脚点外推；室内默认不限，天花板自然挡住；
- **水平在可走范围的投影内**（闭运算 + 填洞），可以再往里收 `--cam-margin`，默认 0；
- **只保留和角色眼睛（落脚点以上 `--cam-eye`，默认 1.62 m）连通的部分**，相机到不了的空腔不要。

输出：
- `camera.ply`：相机体积的边界，平滑后 QEM 减面（相机贴着它滑，没有台阶要跨，所以不用体素面）。在引擎里只挡相机通道：UE 用 Camera channel，Unity 用单独的 layer。
- `camera_sdf.npz`：到边界的有符号距离（米，里面为正），存的是体积周围 ±2 m 的裁剪块。
- `voxels.npz` / `stages.npz` 里多一层 `camera`。
- 调试图：`debug/xsection_camera.png`（过种子点的两个竖直剖面；绿色是相机体积，黄色是可走地面）。

不用引擎时用 `camera_volume.py`（splat 坐标，场景单位）：

```python
from camera_volume import CameraVolume
vol = CameraVolume("out/final_vox")
vol.sdf(p)               # 到边界的距离（米），> 0 在里面
vol.clamp(p)             # 自由相机：外面的点推回到最近的允许位置
vol.arm(pivot, want)     # 第三人称：从头部到期望机位，出体积前截断
```

```bat
py.bat camera_volume.py out/final_vox     :: 随机弹簧臂 + 随机自由相机 + 近处遮挡统计，渲染图在 debug/camera_check.jpg
```

`final.ply`（0.1 m 体素）上的结果，生成多花约 5 s：

| | |
|---|---|
| 相机体积 | 1936 m³（0.3 m 间隙后 5220，限高 5 m 后 1936），`camera.ply` 1.2 万三角形，`camera_sdf.npz` 5.4 MB |
| 角色眼睛在体积内 | 96.3% |
| 第三人称，2000 个随机 3 m 弹簧臂 | 期望机位有 32% 在体积外；39% 被截短，平均 2.30 m；截断后全部在体积内 |
| 自由相机，网格内均匀撒 2000 点 | 78% 在体积外；夹回后全部在体积内，中位移动 2.5 m |
| 被 splat 包住（0.6 m 内整个球面平均不透明度 > 0.3） | 体积内随机点 1.5%（`--cam-opacity 0.1` 时 2.5%）；第三人称机位 0% |

调参的发现：
- **水平收边不能默认开**：`--cam-margin 1` 时只有 85% 的眼睛在体积内，角色走到边上时相机一开始就被挡。
- **间隙只对可见物算**：一开始对"空的部分"的边界也留 0.3 m，眼睛只有 94.7% 在里面；改成只对可见物算，升到 98.7%。
- **`--cam-opacity` 0.1 → 0.03**：被包住的点从 2.5% 降到 1–1.5%，眼睛覆盖从 98.7% 降到 96.3%。`--cam-clear 0.5` 能降到 0%，但眼睛覆盖只剩 90.9%。

局限：
- **只约束位置，不约束朝向。** 这个 splat 没有天空，往上看就是黑的；站在边缘往外看，也会看到重建范围外面。要彻底避免，需要加天空球或雾，或者限制俯仰角。
- 约 4% 的眼睛位置贴着低矮的枝叶，离可见物不到 0.3 m，在体积外。角色站在那里时，相机会贴着头。

#### UE 实测

有 `camera.ply` 的运行，`export_ue.py` 会多导出一层 `CameraVolume`。`ue/setup_level.py` 把它设成只做查询、只挡 Camera 通道，其他通道（Pawn、Visibility）都忽略。试走多一轮：`voxel` 只有外壳，`camera` 外壳加相机层。

```bat
py.bat export_ue.py out/final_vox
:: UE：setup_level.py、walk_test.py，命令同上
py.bat ue/camera_eval.py out/final_vox
```

`walk_test.py` 让每个角色自己的 CameraBoom 照常工作：模板设置为臂长 4 m、12 cm 探测球、Camera 通道。旋转直接在世界坐标系里设：偏航是行走方向 ±60°，俯仰 -50° 到 +10°，每 2–5 s 换一次。每个角色每秒记 4 次臂起点、相机位置和朝向。`camera_eval.py` 把这些位置换回 splat 坐标，拿相机体积检查，再用 gsplat 渲染相机看到的画面。

`final_vox`，64 个角色 × 45 s × 2 轮：

| | 只有外壳 | 外壳 + 相机层 |
|---|---|---|
| 相机在相机体积外 | 30.6%（超出 0.2 m 以上：16.1%） | 4.2%（1.3%；臂起点在体积内时只有 0.6%） |
| 被 splat 包住（0.6 m 内球面平均不透明度 > 0.3） | 14.9% | 1.0% |
| 弹簧臂被截短 | 43.7%，平均臂长 3.06 m | 58.0%，平均 2.63 m |
| 角色行走：效率 / 掉出世界 / 被困 | 0.50 / 0 / 25 | 0.51 / 0 / 25 |

- **相机层不影响角色。** 两轮的行走数据一样。
- **UE 和 Python 对得上。** 同一起点、同一方向，UE 弹簧臂和 `camera_volume.arm()` 的臂长中位差 -2.5 cm，91% 差在 25 cm 以内。UE 用的是 12 cm 探测球。
- **剩下的 4.2% 主要来自臂起点在体积外的情况。** 臂起点是胶囊中心，离地约 0.9 m，约 7% 落在体积外。从封闭网格内部开始扫，有时会直接穿出去。
- **渲染图在 `debug/` 下。**
  - `ue_camera_pairs.jpg`：左边是只有外壳时最糊的 5 个 UE 机位，基本都在树冠里，画面全黑或过曝；右边是同一根臂被相机体积截断后的画面。
  - `ue_camera_views.jpg`：相机层这一轮里臂被截短 1 m 以上的 UE 机位。
- **镜头很低、看地面时仍然糊。** 那是 splat 本身近看的分辨率不够，不是相机进了东西里。

## 体素化过程视频（`video/voxel_video.py`）

把一个场景"一步步变成方块碰撞体"的讲解视频，不需要引擎：

```bat
py.bat voxel_collision.py D:/3DGS_Data/final.ply --out out/final_vox --mpu 0.2 --center 0 -4 --size 24 24 --voxel 0.1 --mode outdoor --regions seed --save-stages 1
py.bat video/voxel_video.py out/final_vox --still 9,26,35,44 --scale 0.5     :: 先看几张预览
py.bat video/voxel_video.py out/final_vox                                    :: 1080p、30 fps、47 s
```

- splat 用 gsplat 渲染；体素用 GPU 光线步进（3D DDA）直接对体素格子求交，每条射线最多穿过 3 个实心格，半透明底色加亮边线，再叠一层光晕。
- 分段：0 s 原始场景 → 4 s 体素化（扫描面经过处，按密度生成方块，前沿带随机抖动）→ 16 s 清理浮游物（红）→ 21 s 可走区域（绿，从起点按步行连通扩散）→ 31 s 碰撞外壳（青，从地面长出，展示到离地 8 m）→ 41 s 恢复原画面，叠加半透明碰撞体。
- 镜头是绕中心的环绕，`--orbit` / `--arc` 可调；脚本会检查镜头是否进了实心。`final.ply` 用的是树冠以上、半径 28、离地 36 个单位的路径。
- `voxel_collision.py --save-stages 1` 会额外写 `stages.npz`（原始实心、清理后、补地面、可站立、可走、空、外壳），视频就从这里读。
- 编码用 `imageio-ffmpeg` 自带的 ffmpeg（libx264）；OpenCV 自带的 H.264 编码器在这台机器上缺 openh264 DLL。
- `final.ply` 的尺度 `--mpu 0.2` 是目测的（草地上的石牌和视角高度），视频里写的"10 cm 方格""身高 1.8 m"都以此为准。

## 展示视频（`video/showcase.py`、`ue/run_capture.py`、`ue/run_video.py`）

```bat
py.bat video/showcase.py out/final_vox bake       :: 体素颜色：从约 290 个视角渲染 splat，反投影到第一个实心体素 -> voxel_rgb.npz
py.bat video/showcase.py out/final_vox orbit      :: 体素化全视角：绕雕像 360°，低 -> 8 m 高 -> 低，24 s
py.bat video/showcase.py out/final_vox wide       :: 大视角：俯瞰整块 24 m 区域，沿雕像中线一半 3DGS、一半体素，24 s
py.bat video/showcase.py out/final_vox split      :: 屏幕分割：分割线左右扫动，左 3DGS、右体素，20 s
py.bat video/showcase.py out/final_vox compare    :: 2×2 对比：3DGS | 体素 | 角色碰撞外壳 | 相机体积，20 s
... --still 5 --scale 2                           :: 单帧（4K）
```
- 体素光线步进是一个 CuPy CUDA 内核（每帧 < 0.1 s），着色（面光照、环境光遮蔽、边线、2× 超采样）在 torch 里。
- `orbit` 的半径按视线自动收缩：从相机到雕像之间如果有雕像以外的体素挡着，就拉近。

### 演示视频：过程 + UE 实机（`video/process_video.py`、`video/montage.py`）

体素化的用途是两个：给角色碰撞，以及不让相机出镜。演示视频就按这个讲，约 46 s：

| 段落 | 步骤 |
|---|---|
| 一 · 场景体素化（10 s） | 3DGS 原场景 → 01 体素化（扫描面经过处出现带 splat 颜色的方块）→ 02 清理浮游物 |
| 二 · 角色碰撞区域（11 s） | 03 可走区域（胶囊体从起点按步行连通扩散）→ 04 角色碰撞外壳（从地面长出，洞被封住、边界成墙） |
| 三 · 相机区域（13.5 s，剖面） | 05 角色能到的空气 → 06 离可见物体 0.3 m 挖掉（红）→ 07 限高 5 m（红色顶部下沉）→ 08 相机体积 |
| 四 · UE 实机（11.6 s） | `ue_run.mp4`：前 11 s 2× 速，撞墙前后原速 |

```bat
py.bat voxel_collision.py D:/3DGS_Data/final.ply --out out/final_vox_stages --mpu 0.2 --center 0 -4 --size 24 24 --voxel 0.1 --mode outdoor --regions seed --save-stages 1
py.bat video/showcase.py out/final_vox_stages bake
py.bat video/process_video.py out/final_vox_stages          :: -> out/final_vox_stages/showcase/process.mp4（34.5 s）
py.bat video/montage.py out/final_vox_stages/showcase/process.mp4 out/final_vox/showcase/ue_run.mp4 out/final_vox/showcase/montage.mp4
```
- `--save-stages 1` 现在还存相机体积的中间步骤 `cam_clear`（离物体 0.3 m 后）、`cam_foot`（限在可走范围上方后）、`cam_cap`（限高后）。
  场景范围和"视线到不了的碎块"两步在 final.ply 上只去掉 30 m³ 和 1 m³，视频里并进第 08 步的字幕。
- 相机段是剖面：相机和雕像之间的半个世界（splat 和体素一起）被一个竖直平面切掉，平面随相机转，能看到体积的高度和物体周围留出的空隙。
- 过程视频用单独重跑的 `out/final_vox_stages`：地面高度估计有微小抖动，重跑后网格整体偏了约 6 mm，体积和可走面积不变；
  UE 那段用的仍是 `out/final_vox`（已导入 UE 的版本）。
- 光线步进在 `video/vox_render.py`（CuPy CUDA 内核做 3D DDA，torch 做着色）。

前一版合集（体素化过程 + 全视角 + 大视角 + 分屏 + 对比）的素材仍可用 `showcase.py` 生成，但已不进演示视频。

UE 单人跑动（`showcase/ue_run.mp4`）：
```powershell
& $ue D:\3DGS_Research\collision_ue\CollisionWalk.uproject -ExecutePythonScript="D:/3DGS_Research/collision_proto/ue/run_capture.py D:/3DGS_Research/collision_proto/out/final_vox/ue" -unattended "-ini:Engine:[/Script/Engine.Engine]:bUseFixedFrameRate=True,[/Script/Engine.Engine]:FixedFrameRate=30.0"
```
然后 `py.bat ue/run_video.py out/final_vox`。
- UE 5.8 没有 3DGS 渲染插件，所以画面是合成的：角色移动、跳跃、弹簧臂相机都是 UE 在关卡里实时算的（体素外壳 + 相机体积）；
  每帧用 SceneCapture（只渲染角色）导出深度、底色和法线，3DGS 用 gsplat 按 UE 相机同一位姿/90° FOV 渲染，按深度遮挡合成，
  加一个接地软阴影。角色光照在合成时按关卡太阳方向重算。
- 路线：绕雕像 5.5 m 跑两圈（中途跳一次），再往外冲到体素外壳边界被挡住（y = 11.4 m，即外壳 11.75 m 减胶囊半径），跳一下再回来。
- 坑：`-UseFixedTimeStep -FPS=30` 对编辑器 PIE 不生效，要用上面的 `-ini:` 打开引擎固定帧率；
  这台机器上 SceneCapture 的 SceneColor / FinalColor 输出全黑（深度正常），所以改采 GBuffer 底色和法线自己打光；
  cv2 读不了 UE 导出的 EXR，用 `OpenEXR` 模块。

## 已知局限

- 这些数字衡量的是**融合结果和 splat 自己的深度是否一致**，不是和真实几何比。没有真值。
- 虚拟相机离训练视角越远，渲染越差。重建区外围、以及这个 splat 的地面视角都不可用。
- 堆场这类细碎结构（桁架、脚手架）融合出来是破碎的。
- 兜底网在重建差的地方会比真实地面低 1–3 m，掉上去是个坑。
- 稠密 TSDF 一个区块上限约 1.4 亿体素（100 m × 100 m × 30 m @ 0.15 m）。更大的场景要分块跑，块间拼接还没做。
- TSDF 路线的减面仍是顶点聚类，三角形大小均匀；`fast-simplification` 现在已装，只用在体素路线上。
- `--mpu 58` 是从卡车宽度和车道宽度目测的，没有控制点。
- UE 里只测了自动随机行走，没有人手动走过；没在 SceneFarm（UE 5.5）里试。
- 补丁地面是按这一块场景调的（离群 0.3 m、补洞 5 m、中值 5×5），换场景要重看 `debug/ground_patch.png` 和试走结果。
- 兜底网 v2 在地面 10 m 内可能浮在更低的真实地面之上（例如往下的坡），是隐形的地板。
- 实际用时建议在游戏逻辑里再加一条：角色站到兜底网上或长时间卡住，就送回上一个安全位置。
- 这里编译出的 gsplat 只保证普通 3DGS 前向渲染可用（见 `build_gsplat.py`）。
