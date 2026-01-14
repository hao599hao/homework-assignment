#include "Astar_searcher.h"

using namespace std;
using namespace Eigen;

void Astarpath::begin_grid_map(double _resolution, Vector3d global_xyz_l,
                                  Vector3d global_xyz_u, int max_x_id,
                                  int max_y_id, int max_z_id) {
  gl_xl = global_xyz_l(0);
  gl_yl = global_xyz_l(1);
  gl_zl = global_xyz_l(2);

  gl_xu = global_xyz_u(0);
  gl_yu = global_xyz_u(1);
  gl_zu = global_xyz_u(2);

  GRID_X_SIZE = max_x_id;
  GRID_Y_SIZE = max_y_id;
  GRID_Z_SIZE = max_z_id;
  GLYZ_SIZE = GRID_Y_SIZE * GRID_Z_SIZE;
  GLXYZ_SIZE = GRID_X_SIZE * GLYZ_SIZE;

  resolution = _resolution;
  inv_resolution = 1.0 / _resolution;

  data = new uint8_t[GLXYZ_SIZE];
  memset(data, 0, GLXYZ_SIZE * sizeof(uint8_t));

  data_raw = new uint8_t[GLXYZ_SIZE];
  memset(data_raw, 0, GLXYZ_SIZE * sizeof(uint8_t));

  Map_Node = new MappingNodePtr **[GRID_X_SIZE];
  for (int i = 0; i < GRID_X_SIZE; i++) {
    Map_Node[i] = new MappingNodePtr *[GRID_Y_SIZE];
    for (int j = 0; j < GRID_Y_SIZE; j++) {
      Map_Node[i][j] = new MappingNodePtr[GRID_Z_SIZE];
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        Vector3i tmpIdx(i, j, k);
        Vector3d pos = gridIndex2coord(tmpIdx);
        Map_Node[i][j][k] = new MappingNode(tmpIdx, pos);
      }
    }
  }
}

void Astarpath::resetGrid(MappingNodePtr ptr) {
  ptr->id = 0;
  ptr->Father = NULL;
  ptr->g_score = inf;
  ptr->f_score = inf;
}

void Astarpath::resetUsedGrids() {
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++)
        resetGrid(Map_Node[i][j][k]);
}

void Astarpath::set_barrier(const double coord_x, const double coord_y,
                             const double coord_z) {
  if (coord_x < gl_xl || coord_y < gl_yl || coord_z < gl_zl ||
      coord_x >= gl_xu || coord_y >= gl_yu || coord_z >= gl_zu)
    return;

  int idx_x = static_cast<int>((coord_x - gl_xl) * inv_resolution);
  int idx_y = static_cast<int>((coord_y - gl_yl) * inv_resolution);
  int idx_z = static_cast<int>((coord_z - gl_zl) * inv_resolution);

  data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;

  if (idx_x == 0 || idx_y == 0 || idx_z == GRID_Z_SIZE || idx_x == GRID_X_SIZE ||
      idx_y == GRID_Y_SIZE)
    data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;
  else {
    data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x + 1) * GLYZ_SIZE + (idx_y + 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x + 1) * GLYZ_SIZE + (idx_y - 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x - 1) * GLYZ_SIZE + (idx_y + 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x - 1) * GLYZ_SIZE + (idx_y - 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x)*GLYZ_SIZE + (idx_y + 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x)*GLYZ_SIZE + (idx_y - 1) * GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x + 1) * GLYZ_SIZE + (idx_y)*GRID_Z_SIZE + idx_z] = 1;
    data[(idx_x - 1) * GLYZ_SIZE + (idx_y)*GRID_Z_SIZE + idx_z] = 1;
  }
}

vector<Vector3d> Astarpath::getVisitedNodes() {
  vector<Vector3d> visited_nodes;
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        // if(Map_Node[i][j][k]->id != 0) // visualize all nodes in open and
        // close list
        if (Map_Node[i][j][k]->id ==
            -1) // visualize nodes in close list only
          visited_nodes.push_back(Map_Node[i][j][k]->coord);
      }

  ROS_WARN("visited_nodes size : %ld", visited_nodes.size());
  return visited_nodes;
}

Vector3d Astarpath::gridIndex2coord(const Vector3i &index) {
  Vector3d pt;

  pt(0) = ((double)index(0) + 0.5) * resolution + gl_xl;
  pt(1) = ((double)index(1) + 0.5) * resolution + gl_yl;
  pt(2) = ((double)index(2) + 0.5) * resolution + gl_zl;

  return pt;
}

Vector3i Astarpath::coord2gridIndex(const Vector3d &pt) {
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);

  return idx;
}

Vector3i Astarpath::c2i(const Vector3d &pt) {
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);

  return idx;
}

Eigen::Vector3d Astarpath::coordRounding(const Eigen::Vector3d &coord) {
  return gridIndex2coord(coord2gridIndex(coord));
}

inline bool Astarpath::isOccupied(const Eigen::Vector3i &index) const {
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy(const Eigen::Vector3i &index) {
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy_raw(const Eigen::Vector3i &index) {
  int idx_x = index(0);
  int idx_y = index(1);
  int idx_z = index(2);
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const Eigen::Vector3i &index) const {
  return isFree(index(0), index(1), index(2));
}

inline bool Astarpath::isOccupied(const int &idx_x, const int &idx_y,
                                        const int &idx_z) const {
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const int &idx_x, const int &idx_y,
                                    const int &idx_z) const {
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] < 1));
}

inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                          vector<MappingNodePtr> &neighborPtrSets,
                                          vector<double> &edgeCostSets) {
  neighborPtrSets.clear();
  edgeCostSets.clear();
  Vector3i Idx_neighbor;
  
  for (int dx = -1; dx < 2; dx++) {
    for (int dy = -1; dy < 2; dy++) {
      for (int dz = -1; dz < 2; dz++) {

        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        Idx_neighbor(0) = (currentPtr->index)(0) + dx;
        Idx_neighbor(1) = (currentPtr->index)(1) + dy;
        Idx_neighbor(2) = (currentPtr->index)(2) + dz;

        if (Idx_neighbor(0) < 0 || Idx_neighbor(0) >= GRID_X_SIZE ||
            Idx_neighbor(1) < 0 || Idx_neighbor(1) >= GRID_Y_SIZE ||
            Idx_neighbor(2) < 0 || Idx_neighbor(2) >= GRID_Z_SIZE) {
          continue;
        }

        neighborPtrSets.push_back(
            Map_Node[Idx_neighbor(0)][Idx_neighbor(1)][Idx_neighbor(2)]);
        edgeCostSets.push_back(sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
  }
}

double Astarpath::getHeu(MappingNodePtr node1, MappingNodePtr node2) {
  if (!node1 || !node2) return std::numeric_limits<double>::max();
  
  // 使用加权欧式距离 + 方向一致性启发式
  double euclidean_dist = (node1->coord - node2->coord).norm();
  
  // 方向一致性启发式（减少拐弯）
  double direction_heuristic = 0.0;
  if (node1->Father) {
    // 计算父节点到当前节点的方向向量
    Vector3d dir_to_current = (node1->coord - node1->Father->coord);
    if (dir_to_current.norm() > 0.01) {  // 避免除以零
      // 计算当前节点到目标节点的方向向量
      Vector3d dir_to_goal = (node2->coord - node1->coord);
      
      // 归一化
      dir_to_current.normalize();
      if (dir_to_goal.norm() > 0.01) {
        dir_to_goal.normalize();
        
        // 计算方向一致性（点积越大，方向越一致）
        double direction_similarity = dir_to_current.dot(dir_to_goal);
        
        // 增加方向一致性启发项
        direction_heuristic = -0.1 * (direction_similarity + 1.0);
      }
    }
  }
  
  // 添加较小的tie_breaker避免路径对称
  unsigned int idx1 = static_cast<unsigned int>(node1->index(0));
  unsigned int idx2 = static_cast<unsigned int>(node1->index(1));
  unsigned int idx3 = static_cast<unsigned int>(node1->index(2));
  // 使用哈希函数生成更小的tie_breaker
  unsigned int hash = (idx1 * 73856093u) ^ (idx2 * 19349663u) ^ (idx3 * 83492791u);
  double tie_breaker = 1e-6 * (hash % 1000u);
  
  // 最终启发值 = 欧式距离 + 方向启发式 + tie_breaker
  return euclidean_dist + direction_heuristic + tie_breaker;
}

bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt) {
  ros::Time time_1 = ros::Time::now();

  // start_point 和 end_point 索引
  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx = coord2gridIndex(end_pt);
  goalIdx = end_idx;

  //start_point 和 end_point 的位置
  start_pt = gridIndex2coord(start_idx);
  end_pt = gridIndex2coord(end_idx);

  // 初始化 struct MappingNode 的指针，分别代表 start node 和 goal node
  // 
  MappingNodePtr startPtr = new MappingNode(start_idx, start_pt);
  MappingNodePtr endPtr = new MappingNode(end_idx, end_pt);

  // Openset 是通过 STL 库中的 multimap 实现的open_list
  Openset.clear();
  // currentPtr 表示 open_list 中 f（n） 最低的节点
  MappingNodePtr currentPtr = NULL;
  MappingNodePtr neighborPtr = NULL;

  // 将 Start 节点放在 Open Set 中
  startPtr->g_score = 0;
  /**
   *
   * STEP 1.1:  完成 Astarpath::getHeu
   *
   * **/
  startPtr->f_score = getHeu(startPtr, endPtr);

  

  startPtr->id = 1;
  startPtr->coord = start_pt;
  startPtr -> Father = NULL;
  Openset.insert(make_pair(startPtr->f_score, startPtr));


  double tentative_g_score;
  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  /**
   *
   * STEP 1.2:  完成循环
   *
   * **/

  while (!Openset.empty()) {
    //1.弹出g+h最小的节点
    //????
    auto iter = Openset.begin(); // 获取OpenSet中f_score最小的节点（multimap首元素）
    currentPtr = iter->second;   // 取出节点指针
    Openset.erase(iter);        // 从OpenSet中删除该节点
    //2.判断是否是终点
    //????
    if (currentPtr->index == goalIdx) { // 当前节点索引等于终点索引
      terminatePtr = currentPtr;       // 标记终止节点
      ros::Time time_2 = ros::Time::now();
      ROS_INFO("Astar search success! Time consume: %f s", (time_2 - time_1).toSec());
      return true; // 搜索成功，返回true
    }
    // 将当前节点标记为已访问（加入CloseSet）
    currentPtr->id = -1;
    //3.拓展当前节点
    //????
    AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets); // 调用邻域扩展函数

    for(unsigned int i=0;i<neighborPtrSets.size();i++)
    {
      
      if(neighborPtrSets[i]->id==-1)
      {
         continue;
      }
      tentative_g_score=currentPtr->g_score+edgeCostSets[i];
      neighborPtr=neighborPtrSets[i];
      if(isOccupied(neighborPtr->index))
      continue;
      if(neighborPtr->id==0)
      {
        //4.填写信息，完成更新
        //???
        neighborPtr->g_score = tentative_g_score;                  // 更新g_score
        neighborPtr->f_score = neighborPtr->g_score + getHeu(neighborPtr, endPtr); // 更新f_score
        neighborPtr->Father = currentPtr;                          // 记录父节点
        neighborPtr->id = 1;                                       // 标记为在OpenSet中
        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr)); // 加入OpenSet
        continue;
      }
      else if(neighborPtr->id==1)
      {
        //???
        if (tentative_g_score < neighborPtr->g_score) {
          // 使用equal_range进行更高效的查找
          auto range = Openset.equal_range(neighborPtr->f_score);
          for (auto it = range.first; it != range.second; ++it) {
            if (it->second == neighborPtr) {
              Openset.erase(it);
              break;
            }
          }
          // 更新代价和父节点
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->f_score = neighborPtr->g_score + getHeu(neighborPtr, endPtr);
          neighborPtr->Father = currentPtr;
          // 重新插入OpenSet
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      continue;
      }
    }
  }

  ros::Time time_2 = ros::Time::now();
  if ((time_2 - time_1).toSec() > 0.1)
    ROS_WARN("Time consume in Astar path finding is %f",
             (time_2 - time_1).toSec());
  return false;
}

vector<Vector3d> Astarpath::getPath() {
  vector<Vector3d> path;
  if (!terminatePtr) {
    ROS_WARN("No path found!");
    return path;
  }
  
  // 从终点回溯到起点
  vector<MappingNodePtr> temp_path;
  MappingNodePtr ptr = terminatePtr;
  
  while (ptr != nullptr) {
    // 确保坐标是最新的
    ptr->coord = gridIndex2coord(ptr->index);
    temp_path.push_back(ptr);
    ptr = ptr->Father;
  }
  /**
   *
   * STEP 1.3:  追溯找到的路径
   *
   * **/

  // ???
  for (auto it = temp_path.rbegin(); it != temp_path.rend(); ++it) {
    path.push_back((*it)->coord);
  }
  if (path.size() > 1) {
    ROS_INFO("Path found with %lu nodes", path.size());
    ROS_INFO("Start: (%.2f, %.2f, %.2f)", path.front()(0), path.front()(1), path.front()(2));
    ROS_INFO("End: (%.2f, %.2f, %.2f)", path.back()(0), path.back()(1), path.back()(2));
  }
  // 输出路径长度日志
  ROS_INFO("Path length: %lu nodes", path.size());

  return path;
}

std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                               double path_resolution) {

  //init
  double dmax=0,d;
  int index=0;
  int end = path.size();
  //1.计算距离首尾连成直线最大的点，并将点集从此处分开
  for(int i=1;i<end-1;i++)
  {
    d=perpendicularDistance(path[i],path[0],path[end-1]);
    if(d>dmax)
    {
      index=i;
      dmax=d;
    }
  }
  vector<Vector3d> subPath1;
  int j = 0;
  while(j<index+1){
    subPath1.push_back(path[j]);
    j++;
  }
  vector<Vector3d> subPath2;
   while(j<int(path.size())){
    subPath2.push_back(path[j]);
    j++;
  }
  //2.拆分点集
  vector<Vector3d> recPath1;
  vector<Vector3d> recPath2;
  vector<Vector3d> resultPath;
  if(dmax>path_resolution)
  {
    recPath1=pathSimplify(subPath1,path_resolution);
    recPath2=pathSimplify(subPath2,path_resolution);
   for(int i=0;i<int(recPath1.size());i++){
    resultPath.push_back(recPath1[i]);
  }
     for(int i=0;i<int(recPath2.size());i++){
    resultPath.push_back(recPath2[i]);
  }
  }else{
    if(path.size()>1){
      resultPath.push_back(path[0]);
      resultPath.push_back(path[end-1]);
    }else{
      resultPath.push_back(path[0]);
    }
    
  }

  return resultPath;
}


double Astarpath::perpendicularDistance(const Eigen::Vector3d point_insert,const Eigen:: Vector3d point_st,const Eigen::Vector3d point_end)
{
  Vector3d line1=point_end-point_st;
  Vector3d line2=point_insert-point_st;
  return double(line2.cross(line1).norm()/line1.norm());
}

Vector3d Astarpath::getPosPoly(MatrixXd polyCoeff, int k, double t) {
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0)
        time(j) = 1.0;
      else
        time(j) = pow(t, j);

    ret(dim) = coeff.dot(time);
    // cout << "dim:" << dim << " coeff:" << coeff << endl;
  }

  return ret;
}


int Astarpath::safeCheck(MatrixXd polyCoeff, VectorXd time) {
  int unsafe_segment = -1; //-1 -> the whole trajectory is safe

  double delta_t=resolution/1.0;//conservative advance step size;
  double t = delta_t;
  Vector3d advancePos;
  for(int i=0;i<polyCoeff.rows();i++)
  {
    while(t<time(i)){
     advancePos=getPosPoly(polyCoeff,i,t) ;
     if(isOccupied(coord2gridIndex(advancePos))){
       unsafe_segment=i;
       break;
     }   
     t+=delta_t;
    }
    if(unsafe_segment!=-1){

      break;
    }else{
      t=delta_t;
    }
  }
  return unsafe_segment;
}

void Astarpath::resetOccupy(){
  for (int i = 0; i < GRID_X_SIZE; i++)
for (int j = 0; j < GRID_Y_SIZE; j++)
  for (int k = 0; k < GRID_Z_SIZE; k++) {
    data[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
    data_raw[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
  }
}