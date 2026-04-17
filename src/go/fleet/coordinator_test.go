package fleet

import (
	"net"
	"net/rpc"
	"testing"
)

func TestFleetCoordination(t *testing.T) {
	robots := []RobotState{
		{ID: "r1", Position: Vec2{X: 0, Y: 0}, Battery: 0.9, Alive: true},
		{ID: "r2", Position: Vec2{X: 5, Y: 5}, Battery: 0.8, Alive: true},
	}
	tasks := []Task{
		{ID: "t1", Target: Vec2{X: 1, Y: 0}},
		{ID: "t2", Target: Vec2{X: 4, Y: 5}},
	}
	if !CompleteTasks(robots, tasks) {
		t.Fatalf("expected all tasks to be assigned")
	}
	assignments := AssignTasks(robots, tasks)
	if assignments["r1"].ID != "t1" || assignments["r2"].ID != "t2" {
		t.Fatalf("expected optimal assignment, got %+v", assignments)
	}
}

func TestGrpcCommunication(t *testing.T) {
	service := NewFleetService()
	server := rpc.NewServer()
	err := server.RegisterName("FleetService", service)
	if err != nil {
		t.Fatalf("failed to register RPC server: %v", err)
	}
	serverConn, clientConn := net.Pipe()
	defer serverConn.Close()
	defer clientConn.Close()
	go server.ServeConn(serverConn)

	client := rpc.NewClient(clientConn)
	defer client.Close()

	registerReply := RegisterReply{}
	err = client.Call("FleetService.RegisterState", RegisterArgs{
		State: RobotState{ID: "r1", Position: Vec2{X: 0, Y: 0}, Battery: 0.8, Alive: true},
	}, &registerReply)
	if err != nil || !registerReply.Accepted {
		t.Fatalf("expected robot registration to succeed, err=%v", err)
	}

	commandReply := CommandReply{}
	err = client.Call("FleetService.SendCommand", CommandArgs{
		Command: FleetCommand{RobotID: "r1", Action: "report", Target: Vec2{X: 0, Y: 0}},
	}, &commandReply)
	if err != nil || !commandReply.Delivered {
		t.Fatalf("expected command delivery to succeed, err=%v", err)
	}

	snapshot := SnapshotReply{}
	err = client.Call("FleetService.Snapshot", SnapshotArgs{}, &snapshot)
	if err != nil {
		t.Fatalf("expected snapshot call to succeed, err=%v", err)
	}
	if len(snapshot.Robots) != 1 || len(snapshot.Commands) != 1 {
		t.Fatalf("expected snapshot to include robot and command, got robots=%d commands=%d", len(snapshot.Robots), len(snapshot.Commands))
	}
}

func TestConsensus(t *testing.T) {
	leader := ElectLeader([]RobotState{
		{ID: "r1", Battery: 0.7, Alive: true},
		{ID: "r2", Battery: 0.9, Alive: true},
		{ID: "r3", Battery: 0.8, Alive: false},
	})
	if leader != "r2" {
		t.Fatalf("expected r2 to become leader, got %s", leader)
	}

	consensus := NewConsensusState()
	consensus.RegisterRobot(RobotState{ID: "r1", Battery: 0.7, Alive: true})
	consensus.RegisterRobot(RobotState{ID: "r2", Battery: 0.9, Alive: true})
	consensus.RegisterRobot(RobotState{ID: "r3", Battery: 0.8, Alive: true})
	consensus.BeginElection("r2")
	consensus.CastVote("r1", "r2")
	consensus.CastVote("r2", "r2")
	if consensus.LeaderID != "r2" {
		t.Fatalf("expected consensus leader r2, got %s", consensus.LeaderID)
	}
}
