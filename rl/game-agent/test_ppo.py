import torch

from train_ppo import PPOTrainer


def test_compute_advantages_and_returns():
    values = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=torch.float32,
    ).T
    rewards = torch.tensor(
        [
            [4, 3, 2, 1],
            [1, 4, 2, 3],
        ],
        dtype=torch.float32,
    ).T
    last_values = torch.tensor([-1, -2], dtype=torch.float32)

    gamma = 0.95
    gae_lambda = 0.8

    dones = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool).T

    # compute reference
    deltas = torch.tensor(
        [
            [4 + gamma * 2 - 1, 3 + gamma * 3 - 2, 2 + gamma * 4 - 3, 1 - 4],
            [1 + gamma * 2 - 1, 4 + gamma * 3 - 2, 2 - 3, 3 + gamma * (-2) - 4],
        ]
    ).T
    fac = gamma * gae_lambda
    delta0 = deltas[:, 0]
    ref_adv0 = torch.tensor(
        [
            delta0[0] + fac * delta0[1] + fac**2 * delta0[2] + fac**3 * delta0[3],
            delta0[1] + fac * delta0[2] + fac**2 * delta0[3],
            delta0[2] + fac * delta0[3],
            delta0[3],
        ]
    )
    delta1 = deltas[:, 1]
    ref_adv1 = torch.tensor(
        [
            delta1[0] + fac * delta1[1] + fac**2 * delta1[2],
            delta1[1] + fac * delta1[2],
            delta1[2],
            delta1[3],
        ]
    )
    ref_advantages = torch.vstack((ref_adv0, ref_adv1)).T
    ref_returns = ref_advantages + values

    out_advantages, out_returns = PPOTrainer.compute_advantages_and_returns(
        values=values, rewards=rewards, last_values=last_values, dones=dones, gamma=gamma, gae_lambda=gae_lambda
    )
    torch.testing.assert_close(out_advantages, ref_advantages)
    torch.testing.assert_close(out_returns, ref_returns)
