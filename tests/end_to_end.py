import unittest
from click.testing import CliRunner
from sca.__main__ import cli, mia, nicv, pearson, sa, ta


class EndToEndTests(unittest.TestCase):
    def test_cli_without_command(self):
        """Test if the CLI behaves correctly without a command"""

        runner = CliRunner()

        result = runner.invoke(cli)

        self.assertFalse(result.exception)
        self.assertEqual("""Usage: cli [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.
""", result.stdout)

    @unittest.skip("Implementation is too slow to run any meaningful tests")
    def test_cli_mia(self):
        """Test if MIA runs after being called from the CLI"""

        runner = CliRunner()

        result = runner.invoke(mia)

        self.assertFalse(result.exception)
        self.assertEqual("""This should perform MIA attack.\n""",
                         result.stdout)

    def test_cli_nicv(self):
        """Test if NICV runs after being called from the CLI"""

        runner = CliRunner()

        result = runner.invoke(nicv, ['--help'])

        self.assertFalse(result.exception)

    def test_cli_pearson(self):
        """Test if pearson runs after being called from the CLI"""

        runner = CliRunner()

        result = runner.invoke(pearson, ['--help'])

        self.assertFalse(result.exception)
        self.assertTrue("Usage: pearson [OPTIONS]" in result.stdout)

    def test_cli_sa(self):
        """Test if SA runs after being called from the CLI"""

        runner = CliRunner()

        result = runner.invoke(sa, ['--help'])

        self.assertFalse(result.exception)
        self.assertTrue("""Usage: sa [OPTIONS]""" in result.stdout)

    def test_cli_ta(self):
        """Test if TA runs after being called from the CLI"""

        runner = CliRunner()

        result = runner.invoke(ta, ['--help'])

        self.assertFalse(result.exception)
        self.assertTrue("""Usage: ta [OPTIONS]""" in result.stdout)

        result = runner.invoke(ta, ['-p', '--help'])

        self.assertFalse(result.exception)
        self.assertTrue("""Usage: ta [OPTIONS]""" in result.stdout)
