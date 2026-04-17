package fleet

import (
	"math"
	"sort"
	"sync"
)

type Vec2 struct {
	X float64
	Y float64
}

type RobotState struct {
	ID       string
	Position Vec2
	Velocity Vec2
	Battery  float64
	Task     string
	Alive    bool
}

type FleetCommand struct {
	RobotID string
	Action  string
	Target  Vec2
}

type Task struct {
	ID     string
	Target Vec2
}

func distance(a Vec2, b Vec2) float64 {
	return math.Hypot(a.X-b.X, a.Y-b.Y)
}

func assignmentCost(robot RobotState, task Task) float64 {
	velocityPenalty := 0.2 * math.Hypot(robot.Velocity.X, robot.Velocity.Y)
	batteryPenalty := (1.0 - robot.Battery) * 5.0
	return distance(robot.Position, task.Target) + velocityPenalty + batteryPenalty
}

func hungarian(cost [][]float64) []int {
	n := len(cost)
	m := len(cost[0])
	u := make([]float64, n+1)
	v := make([]float64, m+1)
	p := make([]int, m+1)
	way := make([]int, m+1)

	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, m+1)
		used := make([]bool, m+1)
		for j := 1; j <= m; j++ {
			minv[j] = math.Inf(1)
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := 0
			for j := 1; j <= m; j++ {
				if used[j] {
					continue
				}
				cur := cost[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= m; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	assignment := make([]int, n)
	for i := range assignment {
		assignment[i] = -1
	}
	for j := 1; j <= m; j++ {
		if p[j] > 0 && p[j] <= n {
			assignment[p[j]-1] = j - 1
		}
	}
	return assignment
}

func AssignTasks(robots []RobotState, tasks []Task) map[string]Task {
	assignments := make(map[string]Task, len(robots))
	if len(robots) == 0 || len(tasks) == 0 {
		return assignments
	}

	n := len(robots)
	m := len(tasks)
	size := n
	if m > size {
		size = m
	}
	cost := make([][]float64, size)
	for i := 0; i < size; i++ {
		cost[i] = make([]float64, size)
		for j := 0; j < size; j++ {
			cost[i][j] = 1e6
		}
	}
	for i, robot := range robots {
		for j, task := range tasks {
			cost[i][j] = assignmentCost(robot, task)
		}
	}
	assignment := hungarian(cost)
	for i, taskIdx := range assignment {
		if i < len(robots) && taskIdx >= 0 && taskIdx < len(tasks) {
			assignments[robots[i].ID] = tasks[taskIdx]
		}
	}
	return assignments
}

func CoordinateFleet(robots []RobotState, tasks []Task) []FleetCommand {
	assignments := AssignTasks(robots, tasks)
	commands := make([]FleetCommand, 0, len(assignments))
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, robot := range robots {
		task, ok := assignments[robot.ID]
		if !ok {
			continue
		}
		wg.Add(1)
		go func(r RobotState, t Task) {
			defer wg.Done()
			command := FleetCommand{RobotID: r.ID, Action: "go_to", Target: t.Target}
			mu.Lock()
			commands = append(commands, command)
			mu.Unlock()
		}(robot, task)
	}

	wg.Wait()
	sort.Slice(commands, func(i, j int) bool { return commands[i].RobotID < commands[j].RobotID })
	return commands
}

func ElectLeader(states []RobotState) string {
	live := make([]RobotState, 0, len(states))
	for _, state := range states {
		if state.Alive {
			live = append(live, state)
		}
	}
	sort.Slice(live, func(i, j int) bool {
		if live[i].Battery == live[j].Battery {
			return live[i].ID < live[j].ID
		}
		return live[i].Battery > live[j].Battery
	})
	if len(live) == 0 {
		return ""
	}
	return live[0].ID
}

func CompleteTasks(robots []RobotState, tasks []Task) bool {
	commands := CoordinateFleet(robots, tasks)
	return len(commands) == len(tasks)
}
