#include "../tests/tests.h"
#include "../search/handicap.h"
#include "../program/setup.h"

#include <limits>

using namespace std;

void Tests::runHandicapMoyoTests() {
  cout << "Running handicap moyo exploration tests" << endl;
  Board board(19,19);
  board.setStone(Location::getLoc(3,3,19), P_BLACK);
  board.setStone(Location::getLoc(15,15,19), P_BLACK);
  Rules rules = Rules::getTrompTaylorish();
  BoardHistory hist(board, P_WHITE, rules, 0, BoardHistoryModes(false,false));
  testAssert(hist.moveHistory.empty());
  testAssert(hist.computeNumHandicapStones() == 2);

  SearchParams params;
  params.complexityBonus = 0.12;
  params.complexityMaxBonus = 0.15;
  Color safeArea[Board::MAX_ARR_SIZE] = {};
  board.calculateArea(safeArea, false, false, false, hist.suicideLegalForPassAlive());

  // Ownership storage deliberately excludes the extra pass-policy entry.
  NNOutput nn;
  nn.nnXLen = NNPos::MAX_BOARD_LEN;
  nn.nnYLen = 19;
  nn.whiteScoreMean = -20.0f;
  const int nnArea = nn.nnXLen * nn.nnYLen;
  nn.whiteOwnerMap = new float[nnArea];
  std::fill(nn.whiteOwnerMap, nn.whiteOwnerMap + nnArea, 0.8f);
  float original[NNPos::MAX_NN_POLICY_SIZE];
  std::fill(original, original + NNPos::MAX_NN_POLICY_SIZE, -1.0f);
  const int passPos = NNPos::getPassPos(nn.nnXLen, nn.nnYLen);
  for(int y = 0; y < 19; y++) {
    for(int x = 0; x < 19; x++) {
      const Loc loc = Location::getLoc(x,y,19);
      const int pos = NNPos::xyToPos(x,y,nn.nnXLen);
      if(hist.isLegal(board,loc,P_WHITE))
        original[pos] = 0.9f / 359.0f;
      if(x >= 7 && x <= 13 && y >= 6 && y <= 12)
        nn.whiteOwnerMap[pos] = -0.6f;
    }
  }
  original[passPos] = 0.1f;
  float policy[NNPos::MAX_NN_POLICY_SIZE];
  auto reset = [&]() { std::copy(original, original + NNPos::MAX_NN_POLICY_SIZE, policy); };
  auto unchanged = [&]() {
    for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
      testAssert(policy[i] == original[i]);
  };
  auto apply = [&]() { return HandicapSearch::applyMoyoPolicy(board,hist,P_WHITE,safeArea,params,nn,policy); };

  reset();
  testAssert(apply());
  const int targetPos = NNPos::xyToPos(10,9,nn.nnXLen);
  testAssert(policy[targetPos] > original[targetPos]);
  testAssert(policy[passPos] == original[passPos]);
  double before = 0.0, after = 0.0;
  for(int i = 0; i <= passPos; i++) {
    if(original[i] < 0.0)
      testAssert(policy[i] == original[i]);
    else {
      before += original[i];
      after += policy[i];
      testAssert(policy[i] <= 3.00001 * original[i]);
    }
  }
  testAssert(std::abs(before - after) < 1e-6);

  // Off is bit-for-bit off; never enable for black, even games, ahead positions,
  // late-game imported positions, encore, finished games, or missing ownership.
  reset();
  params.complexityBonus = 0.0;
  testAssert(!apply()); unchanged();
  params.complexityBonus = 0.12;
  params.complexityMaxBonus = 0.0;
  testAssert(!apply()); unchanged();
  params.complexityMaxBonus = 0.15;
  testAssert(!HandicapSearch::applyMoyoPolicy(board,hist,P_BLACK,safeArea,params,nn,policy)); unchanged();
  params.complexityMinHandicap = 3;
  testAssert(!apply()); unchanged();
  params.complexityMinHandicap = 2;
  BoardHistory evenHist = hist;
  evenHist.overrideNumHandicapStones = 0;
  testAssert(!HandicapSearch::applyMoyoPolicy(board,evenHist,P_WHITE,safeArea,params,nn,policy)); unchanged();
  nn.whiteScoreMean = 1.0f;
  testAssert(!apply()); unchanged();
  nn.whiteScoreMean = -20.0f;
  hist.initialTurnNumber = 221;
  testAssert(!apply()); unchanged();
  hist.initialTurnNumber = 0;
  hist.encorePhase = 1;
  testAssert(!apply()); unchanged();
  hist.encorePhase = 0;
  hist.isGameFinished = true;
  testAssert(!apply()); unchanged();
  hist.isGameFinished = false;
  float* ownerMap = nn.whiteOwnerMap;
  nn.whiteOwnerMap = nullptr;
  testAssert(!apply()); unchanged();
  nn.whiteOwnerMap = ownerMap;

  // Very confident opponent ownership and tiny isolated patches get no budget.
  std::fill(ownerMap, ownerMap + nnArea, -0.999f);
  testAssert(!apply()); unchanged();
  std::fill(ownerMap, ownerMap + nnArea, 0.8f);
  ownerMap[targetPos] = -0.6f;
  testAssert(!apply()); unchanged();

  // Empty point inside a pass-alive region must not gain mass, even when the
  // ownership prediction disagrees. Zero policy entries must stay zero too.
  std::fill(ownerMap, ownerMap + nnArea, -0.6f);
  const Loc targetLoc = Location::getLoc(10,9,19);
  safeArea[targetLoc] = P_BLACK;
  reset();
  testAssert(apply());
  testAssert(policy[targetPos] < original[targetPos]);
  safeArea[targetLoc] = C_EMPTY;
  reset();
  policy[targetPos] = 0.0f;
  testAssert(apply());
  testAssert(policy[targetPos] == 0.0f);

  // Invalid NN data must not introduce NaNs into policy.
  reset();
  nn.whiteScoreMean = std::numeric_limits<float>::quiet_NaN();
  testAssert(!apply()); unchanged();
  nn.whiteScoreMean = -20.0f;
  std::fill(ownerMap, ownerMap + nnArea, std::numeric_limits<float>::quiet_NaN());
  testAssert(!apply()); unchanged();

  // Repeated black moves, as used by some GTP clients, use BoardHistory's
  // existing handicap inference instead of maintaining a second ad-hoc counter.
  Board playedBoard(19,19);
  BoardHistory playedHist(playedBoard,P_BLACK,rules,0,BoardHistoryModes(false,false));
  playedHist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
  playedHist.makeBoardMoveAssumeLegal(playedBoard,Location::getLoc(3,3,19),P_BLACK,nullptr);
  playedHist.makeBoardMoveAssumeLegal(playedBoard,Location::getLoc(15,15,19),P_BLACK,nullptr);
  testAssert(playedHist.computeNumHandicapStones() == 2);
  testAssert(HandicapSearch::explorationBudget(playedBoard,playedHist,P_WHITE,params,nn) > 0.0);

  // A smaller rectangular board in a 19x19 NN buffer: padding and pass untouched.
  {
    Board small(9,7);
    small.setStone(Location::getLoc(2,2,9),P_BLACK);
    small.setStone(Location::getLoc(6,4,9),P_BLACK);
    BoardHistory smallHist(small,P_WHITE,rules,0,BoardHistoryModes(false,false));
    Color smallSafe[Board::MAX_ARR_SIZE] = {};
    small.calculateArea(smallSafe,false,false,false,smallHist.suicideLegalForPassAlive());
    std::fill(ownerMap,ownerMap + nnArea,-0.6f);
    std::fill(policy,policy + NNPos::MAX_NN_POLICY_SIZE,-1.0f);
    for(int y = 0; y < 7; y++) {
      for(int x = 0; x < 9; x++) {
        if(smallHist.isLegal(small,Location::getLoc(x,y,9),P_WHITE))
          policy[NNPos::xyToPos(x,y,nn.nnXLen)] = 0.9f / 61.0f;
      }
    }
    policy[passPos] = 0.1f;
    testAssert(HandicapSearch::applyMoyoPolicy(small,smallHist,P_WHITE,smallSafe,params,nn,policy));
    double sum = 0.0;
    for(int y = 0; y < nn.nnYLen; y++) {
      for(int x = 0; x < nn.nnXLen; x++) {
        const int pos = NNPos::xyToPos(x,y,nn.nnXLen);
        if(x >= 9 || y >= 7)
          testAssert(policy[pos] == -1.0f);
        if(policy[pos] > 0.0f)
          sum += policy[pos];
      }
    }
    testAssert(std::abs(sum + policy[passPos] - 1.0) < 1e-6);
    testAssert(policy[passPos] == 0.1f);
    smallHist.initialTurnNumber = 39;
    testAssert(HandicapSearch::explorationBudget(small,smallHist,P_WHITE,params,nn) == 0.0);
  }

  // Legal self-atari is not an exploration target (not a general life/death test).
  {
    Board atari(19,19);
    atari.setStone(Location::getLoc(9,9,19),P_BLACK);
    atari.setStone(Location::getLoc(11,9,19),P_BLACK);
    atari.setStone(Location::getLoc(10,8,19),P_BLACK);
    BoardHistory atariHist(atari,P_WHITE,rules,0,BoardHistoryModes(false,false));
    Color atariSafe[Board::MAX_ARR_SIZE] = {};
    atari.calculateArea(atariSafe,false,false,false,atariHist.suicideLegalForPassAlive());
    testAssert(atariHist.isLegal(atari,targetLoc,P_WHITE));
    testAssert(atari.getNumLibertiesAfterPlay(targetLoc,P_WHITE,2) == 1);
    reset();
    testAssert(HandicapSearch::applyMoyoPolicy(atari,atariHist,P_WHITE,atariSafe,params,nn,policy));
    testAssert(policy[targetPos] < original[targetPos]);
  }

  // Shared config loader: GTP/analysis and match's indexed settings agree.
  {
    ConfigParser cfg;
    cfg.overrideKey("numSearchThreads","1");
    cfg.overrideKey("complexityBonus","0.08");
    cfg.overrideKey("complexityMinHandicap","3");
    cfg.overrideKey("complexityMaxBonus","0.12");
    const SearchParams gtp = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    const SearchParams analysis = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_ANALYSIS);
    testAssert(gtp.complexityBonus == 0.08 && analysis.complexityBonus == 0.08);
    testAssert(gtp.complexityMinHandicap == 3 && analysis.complexityMinHandicap == 3);
    testAssert(gtp.complexityMaxBonus == 0.12 && analysis.complexityMaxBonus == 0.12);
    cfg.overrideKey("numBots","2");
    cfg.overrideKey("complexityBonus1","0.0");
    cfg.overrideKey("complexityMinHandicap1","5");
    cfg.overrideKey("complexityMaxBonus1","0.2");
    const vector<SearchParams> bots = Setup::loadParams(cfg,Setup::SETUP_FOR_OTHER);
    testAssert(bots.size() == 2);
    testAssert(bots[0].complexityBonus == 0.08 && bots[1].complexityBonus == 0.0);
    testAssert(bots[0].complexityMinHandicap == 3 && bots[1].complexityMinHandicap == 5);
    testAssert(bots[0].complexityMaxBonus == 0.12 && bots[1].complexityMaxBonus == 0.2);
  }

  // Custom parameters participate in equality, reporting, and cache identity.
  SearchParams base;
  for(int i = 0; i < 3; i++) {
    SearchParams changed = base;
    if(i == 0) changed.complexityBonus = 0.1;
    if(i == 1) changed.complexityMaxBonus = 0.1;
    if(i == 2) changed.complexityMinHandicap = 3;
    testAssert(changed != base);
    testAssert(changed.getHash() != base.getHash());
    testAssert(changed.changeableParametersToJson() != base.changeableParametersToJson());
  }
  cout << "Handicap moyo exploration tests passed" << endl;
}
