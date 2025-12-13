package bdd.steps;

import io.cucumber.java.en.Given;


public class LoginSteps {

    @Given("I am logged in as {string} with password {string}")
    public void iAmLoggedIn(String email, String password) {
        SharedClass.driver.get("http://localhost:4200/auth/login");
        SharedClass.loginPage.loginAs(email, password);
    }

    @Given("I am logged in as admin {string} with password {string}")
    public void iAmLoggedInAsAdminWithPassword(String email, String password) {
        SharedClass.driver.get("http://localhost:4200/auth/login");
        SharedClass.loginPage.loginAsAdmin(email, password);
    }

}
