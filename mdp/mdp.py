import numpy as np

class OneDWorld(object):
    """docstring for OneDWorld"""
    def __init__(self, size, start_state=0, terminal_state=None, reward=None):
        super(OneDWorld, self).__init__()
        self._size = size
        self.reward = reward
        self.start_state = start_state
        self.terminal_state = terminal_state
        self._state = start_state

        if terminal_state is None:
            self.terminal_state = [self._size - 1]

        if reward is None:
            self.reward = self.terminal_state


    def step(self, action):
        print('Action {}'.format(action))
        if action == 0:
            self._state -= 1
        elif action == 1:
            self._state += 1
        else:
            print('unknown action.')

        if self._state < 0:
            self._state = 0

        done = False
        if self._state in self.terminal_state:
            # print('Am in terminal state')
            done = True

        r = 0
        if self._state in self.reward:
            r += 1

        return self._state, r, done

    def reset(self):
        self._state = self.start_state

    @property
    def state(self):
        return self._state

    @property
    def size(self):
        return self._size



class Option(object):
    """docstring for Option"""
    def __init__(self, beta, true_action=1, noise=0.3, seed=1):
        super(Option, self).__init__()
        self.beta = beta
        self.noise = noise
        self.true_action = true_action
        self.seed = seed

        np.random.seed(seed)

    def get_action(self, state):
        action = self.true_action
        if action < 0:
            print('Random action selected')
            action = np.random.randint(1)
            print(action)

        terminated = False
        if np.random.uniform() > self.beta:
            terminated = True

        return action, terminated

    def forward_simulate(self, state, n, world_size):
        done = False
        for i in range(n):
            a, terminated = self.get_action(state)
            w = OneDWorld(size = world_size, start_state=state)
            next_state, r, done = w.step(a)

            # If noisy, don't move
            if np.random.uniform() < self.noise:
                # print('noisy state')
                next_state = state

            state = next_state

            if (terminated):
                # print('terminated')
                break
            if done:
                break

        return state, done

    def run_option(self, world):
        state = world.state

        terminated = False
        done = False
        while not terminated and not done:
            a, terminated = self.get_action(state)
            state, reward, done = world.step(a)

        return done


def run_until_complete(options, world, steps=10):
    n = 1

    done = False
    options_used = []
    while not done:
        max_state = world.state

        best_opton = None
        option_id = None
        for i, option in enumerate(options):
            state, done = option.forward_simulate(state=world.state, n=steps, world_size=world.size)

            print('For option {} reached state {}. Done? {}'.format(i, state, done))

            if state > max_state or best_opton is None:
                best_opton = option
                option_id = i
                max_state = state



        print('Best option from {}, to {} with {}'.format(world.state, max_state, option_id))


        # Forward option
        options_used.append(option_id)
        done = best_opton.run_option(world)
        print('Step: {} New world state: {}. Done? {}'.format(n, world.state, done))
        n += 1
        # exit(0)
    print('Options used: ', options_used)

def main():
    world = OneDWorld(size=20)
    good = Option(beta=0.9)
    med = Option(beta=0.5)
    bad = Option(beta=0.2, seed=0)
    random = Option(beta=0.5, seed=1, true_action=-1)
    wrong = Option(beta=0.2, seed=2, true_action=0)

    run_until_complete(options=[good, med, bad, random, wrong], world=world)


if __name__ == '__main__':
    main()
