import subprocess
import sys

import unittest

sys.path.append('../../')
from app.config.logger import fed_logger


class TestFed(unittest.TestCase):
    def test_classic_1_1(self):
        with open("test_config/test_classic_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_classic_1_1.yaml', 'up', '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_classic_1_3(self):
        with open("test_config/test_classic_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_classic_1_3.yaml', 'up', '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fake_offloading_1_1(self):
        with open("test_config/test_fake_offloading_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fake_offloading_1_1.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_rl_offloading_1_3(self):
        with open("test_config/test_rl_offloading_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_rl_offloading_1_3.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fake_offloading_1_3(self):
        with open("test_config/test_fake_offloading_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fake_offloading_1_3.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fake_offloading_1_1_1(self):
        with open("test_config/test_fake_offloading_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fake_offloading_1_1_1.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fake_offloading_1_1_3(self):
        with open("test_config/test_fake_offloading_1_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fake_offloading_1_1_3.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fake_offloading_1_2_4(self):
        with open("test_config/test_fake_offloading_1_2_4_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fake_offloading_1_2_4.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_no_offloading_1_1_1(self):
        with open("test_config/test_no_offloading_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_no_offloading_1_1_1.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_no_offloading_1_1_3(self):
        with open("test_config/test_no_offloading_1_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_no_offloading_1_1_3.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_no_offloading_1_2_4(self):
        with open("test_config/test_no_offloading_1_2_4_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_no_offloading_1_2_4.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_only_edge_offloading_1_2_4(self):
        with open("test_config/test_only_edge_offloading_1_2_4_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_only_edge_offloading_1_2_4.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_random_offloading_1_1_1(self):
        with open("test_config/test_random_offloading_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_random_offloading_1_1_1.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_only_server_offloading_1_2_4(self):
        with open("test_config/test_only_server_offloading_1_2_4_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_only_server_offloading_1_2_4.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_fedmec_offloading_1_1_1(self):
        with open("test_config/test_fedmec_offloading_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fedmec_offloading_1_1_1.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_rl_offloading_1_1_3(self):
        with open("test_config/test_rl_offloading_1_1_3_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_fake_offloading_simnet_1_1_3.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))

    def test_rl_training_1_1_1(self):
        with open("test_config/test_rl_training_1_1_1_config.py", "r") as f:
            data = f.read()

        with open("../../app/config/config.py", "w") as f:
            f.write(data)
        subprocess.run(['docker', 'build', '-t', 'fed-flow:test', '../../'])
        test = subprocess.run(
            ['docker', 'compose', '-f', 'docker_compose/test_rl_training_1_1_1.yaml', 'up',
             '--remove-orphans'])
        subprocess.run(['docker', 'system', 'prune', '-f'])

        fed_logger.info(str(test.stdout))
        if test.returncode != 0:
            self.fail(str(test.stderr))
