package fleet

import (
	"net"
	"net/rpc"
	"sync"
)

type RegisterArgs struct {
	State RobotState
}

type RegisterReply struct {
	Accepted bool
}

type CommandArgs struct {
	Command FleetCommand
}

type CommandReply struct {
	Delivered bool
}

type SnapshotArgs struct{}

type SnapshotReply struct {
	Robots   []RobotState
	Commands []FleetCommand
	Leader   string
}

type FleetService struct {
	mu       sync.Mutex
	robots   map[string]RobotState
	commands []FleetCommand
}

func NewFleetService() *FleetService {
	return &FleetService{
		robots:   make(map[string]RobotState),
		commands: make([]FleetCommand, 0),
	}
}

func (f *FleetService) RegisterState(args RegisterArgs, reply *RegisterReply) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.robots[args.State.ID] = args.State
	reply.Accepted = true
	return nil
}

func (f *FleetService) SendCommand(args CommandArgs, reply *CommandReply) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.commands = append(f.commands, args.Command)
	reply.Delivered = true
	return nil
}

func (f *FleetService) Snapshot(args SnapshotArgs, reply *SnapshotReply) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	reply.Robots = make([]RobotState, 0, len(f.robots))
	for _, robot := range f.robots {
		reply.Robots = append(reply.Robots, robot)
	}
	reply.Commands = append([]FleetCommand(nil), f.commands...)
	reply.Leader = ElectLeader(reply.Robots)
	return nil
}

func StartFleetRPCServer(service *FleetService) (net.Listener, error) {
	server := rpc.NewServer()
	if err := server.RegisterName("FleetService", service); err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	go server.Accept(listener)
	return listener, nil
}
