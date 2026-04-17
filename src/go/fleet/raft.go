package fleet

type ConsensusState struct {
	Term        int
	LeaderID    string
	Votes       map[string]string
	KnownRobots map[string]RobotState
}

func NewConsensusState() *ConsensusState {
	return &ConsensusState{
		Term:        0,
		Votes:       make(map[string]string),
		KnownRobots: make(map[string]RobotState),
	}
}

func (c *ConsensusState) RegisterRobot(robot RobotState) {
	c.KnownRobots[robot.ID] = robot
}

func (c *ConsensusState) BeginElection(candidateID string) {
	c.Term++
	c.Votes = map[string]string{candidateID: candidateID}
	c.LeaderID = ""
}

func (c *ConsensusState) CastVote(voterID string, candidateID string) {
	c.Votes[voterID] = candidateID
	c.recomputeLeader()
}

func (c *ConsensusState) recomputeLeader() {
	counts := map[string]int{}
	totalLive := 0
	for _, robot := range c.KnownRobots {
		if robot.Alive {
			totalLive++
		}
	}
	for _, candidate := range c.Votes {
		counts[candidate]++
		if counts[candidate] > totalLive/2 {
			c.LeaderID = candidate
		}
	}
}
