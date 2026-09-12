#ifndef SEARCH_HANDICAP_H_
#define SEARCH_HANDICAP_H_

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../neuralnet/nninputs.h"
#include "../search/searchparams.h"

// Experimental root-only exploration, NOT an additional value/territory reward.
// Ownership is a prediction, not proof that an invasion works. The normal search
// and final move selection must still be allowed to reject every suggested move.
namespace HandicapSearch {

inline double clamp01(double x) {
  return std::max(0.0, std::min(1.0, x));
}

inline double explorationBudget(
  const Board& board, const BoardHistory& hist, Player pla,
  const SearchParams& params, const NNOutput& nnOutput,
  double neutralScoreMean = NAN, const float* neutralOwnerMap = nullptr
) {
  const float* ownerMap = neutralOwnerMap != nullptr ? neutralOwnerMap : nnOutput.whiteOwnerMap;
  const double scoreMean = std::isfinite(neutralScoreMean) ? neutralScoreMean : nnOutput.whiteScoreMean;
  if(pla != P_WHITE || params.complexityBonus <= 0.0 || params.complexityMaxBonus <= 0.0 ||
     hist.isGameFinished || hist.encorePhase > 0 || ownerMap == nullptr ||
     !std::isfinite(scoreMean) ||
     hist.computeNumHandicapStones() < std::max(2, params.complexityMinHandicap))
    return 0.0;

  // Fade out when ahead and in the late endgame. Use the actual turn number,
  // including an SGF/analysis position's initialTurnNumber, and scale by area.
  const double turn19 = hist.getCurrentTurnNumber() * (361.0 / (board.x_size * board.y_size));
  const double phase = clamp01((220.0 - turn19) / 80.0);
  const double behind = clamp01(-scoreMean / 15.0);
  return std::min(0.25, std::min(params.complexityBonus, params.complexityMaxBonus)) * phase * behind;
}

struct ContestableRegion {
  std::vector<Loc> points;
  int contestablePoints = 0;
  double opponentOwnerSum = 0.0;
};

inline bool isContestableEmpty(const Board& board, const Color* safeArea, Loc loc) {
  return board.colors[loc] == C_EMPTY && safeArea[loc] == C_EMPTY;
}

inline std::vector<ContestableRegion> findContestableRegions(
  const Board& board, const Color* safeArea, const NNOutput& nnOutput, const float* ownerMap
) {
  std::vector<ContestableRegion> regions;
  bool seen[Board::MAX_ARR_SIZE] = {};
  std::vector<Loc> queue;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      const Loc start = Location::getLoc(x,y,board.x_size);
      if(seen[start] || !isContestableEmpty(board,safeArea,start))
        continue;
      ContestableRegion region;
      queue.clear();
      queue.push_back(start);
      seen[start] = true;
      for(size_t i = 0; i < queue.size(); i++) {
        const Loc loc = queue[i];
        region.points.push_back(loc);
        const int xx = Location::getX(loc,board.x_size);
        const int yy = Location::getY(loc,board.x_size);
        const double opponentOwner = -ownerMap[NNPos::xyToPos(xx,yy,nnOutput.nnXLen)];
        if(std::isfinite(opponentOwner) && opponentOwner > 0.10 && opponentOwner < 0.97) {
          region.contestablePoints++;
          region.opponentOwnerSum += clamp01((opponentOwner + 0.10) / 1.10);
        }
        const Loc neighbors[4] = {
          Location::getLoc(xx-1,yy,board.x_size),
          Location::getLoc(xx+1,yy,board.x_size),
          Location::getLoc(xx,yy-1,board.x_size),
          Location::getLoc(xx,yy+1,board.x_size)
        };
        for(const Loc next : neighbors) {
          if(board.isOnBoard(next) && !seen[next] && isContestableEmpty(board,safeArea,next)) {
            seen[next] = true;
            queue.push_back(next);
          }
        }
      }
      regions.push_back(std::move(region));
    }
  }
  return regions;
}

// policy is a PRIVATE copy of root policy, after temperature but before noise.
// safeArea uses Board::calculateArea with all three territory flags false.
// Returns false without modifying policy when there is nothing to do.
inline bool applyMoyoPolicy(
  const Board& board, const BoardHistory& hist, Player pla,
  const Color* safeArea, const SearchParams& params, const NNOutput& nnOutput,
  float* policy, double neutralScoreMean = NAN, const float* neutralOwnerMap = nullptr
) {
  const float* ownerMap = neutralOwnerMap != nullptr ? neutralOwnerMap : nnOutput.whiteOwnerMap;
  const double budget = explorationBudget(board,hist,pla,params,nnOutput,neutralScoreMean,ownerMap);
  if(budget <= 0.0 || safeArea == nullptr || policy == nullptr)
    return false;
  const int nnXLen = nnOutput.nnXLen;
  const int nnYLen = nnOutput.nnYLen;
  if(nnXLen < board.x_size || nnYLen < board.y_size ||
     nnXLen > NNPos::MAX_BOARD_LEN || nnYLen > NNPos::MAX_BOARD_LEN)
    return false;

  std::vector<ContestableRegion> regions = findContestableRegions(board,safeArea,nnOutput,ownerMap);
  double target[NNPos::MAX_NN_POLICY_SIZE] = {};
  double targetSum = 0.0;
  double boardPolicySum = 0.0;
  bool pointInUsefulRegion[Board::MAX_ARR_SIZE] = {};
  double regionFactorByLoc[Board::MAX_ARR_SIZE] = {};
  for(const ContestableRegion& region : regions) {
    if(region.points.size() < 10 || region.contestablePoints < 8)
      continue;
    const double contestableProp = (double)region.contestablePoints / region.points.size();
    const double meanOwner = region.opponentOwnerSum / std::max(1,region.contestablePoints);
    const double regionFactor =
      clamp01((contestableProp - 0.10) / 0.45) * clamp01((meanOwner - 0.18) / 0.45);
    if(regionFactor <= 0.0)
      continue;
    for(const Loc loc : region.points) {
      pointInUsefulRegion[loc] = true;
      regionFactorByLoc[loc] = regionFactor;
    }
  }
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      const int pos = NNPos::xyToPos(x, y, nnXLen);
      const double prior = policy[pos];
      if(!std::isfinite(prior))
        return false;
      if(prior > 0.0)
        boardPolicySum += prior;
      // No artificial floor: do not resurrect illegal, zero-prior, or effectively
      // unconsidered moves. Relative boosts below are also capped at 3x.
      if(prior < 1e-5)
        continue;
      const Loc loc = Location::getLoc(x, y, board.x_size);
      if(board.colors[loc] != C_EMPTY || safeArea[loc] != C_EMPTY)
        continue;
      const double opponentOwner = -ownerMap[pos];
      if(!std::isfinite(opponentOwner) || opponentOwner <= -0.10 || opponentOwner >= 0.97)
        continue;
      if(!hist.isLegal(board, loc, pla) || board.getNumLibertiesAfterPlay(loc, pla, 2) < 2)
        continue;
      if(!pointInUsefulRegion[loc])
        continue;

      // Local 7x7/Manhattan-radius-3 empty-area pressure, not a claim to identify
      // connected territory. Require several nearby unsettled black-leaning
      // empty points so one ownership pixel cannot trigger a "moyo" reward.
      double nearbyPressure = 0.0;
      for(int dy = -3; dy <= 3; dy++) {
        for(int dx = -3; dx <= 3; dx++) {
          if(std::abs(dx) + std::abs(dy) > 3)
            continue;
          const int xx = x + dx;
          const int yy = y + dy;
          if(xx < 0 || xx >= board.x_size || yy < 0 || yy >= board.y_size)
            continue;
          const Loc nearLoc = Location::getLoc(xx, yy, board.x_size);
          if(board.colors[nearLoc] != C_EMPTY || safeArea[nearLoc] != C_EMPTY)
            continue;
          const double owner = -ownerMap[NNPos::xyToPos(xx, yy, nnXLen)];
          if(std::isfinite(owner) && owner > 0.10 && owner < 0.97)
            nearbyPressure += clamp01((owner - 0.10) / 0.40) * clamp01((0.97 - owner) / 0.22);
        }
      }
      const double region = regionFactorByLoc[loc] * clamp01((nearbyPressure - 5.0) / 10.0);
      const double contestable =
        clamp01((opponentOwner + 0.10) / 0.50) * clamp01((0.97 - opponentOwner) / 0.22);
      target[pos] = std::sqrt(prior) * region * contestable;
      targetSum += target[pos];
    }
  }
  if(targetSum <= 0.0 || boardPolicySum <= 0.0)
    return false;

  // Redistribute at most budget of the NON-PASS mass. Pass is unchanged, illegal
  // entries stay negative, and each candidate stays below 3x its original prior.
  double addedSum = 0.0;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      const int pos = NNPos::xyToPos(x, y, nnXLen);
      if(target[pos] > 0.0) {
        target[pos] = std::min(budget * boardPolicySum * target[pos] / targetSum, 2.0 * policy[pos]);
        addedSum += target[pos];
      }
    }
  }
  if(addedSum <= 0.0)
    return false;
  const double retained = 1.0 - addedSum / boardPolicySum;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      const int pos = NNPos::xyToPos(x, y, nnXLen);
      if(policy[pos] > 0.0)
        policy[pos] = static_cast<float>(retained * policy[pos] + target[pos]);
    }
  }
  return true;
}

} // namespace HandicapSearch

#endif
