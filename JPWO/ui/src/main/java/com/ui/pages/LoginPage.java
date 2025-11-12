package com.ui.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class LoginPage {

    private WebDriver driver;
    private WebDriverWait wait;

    private By usernameField = By.id("email");
    private By passwordField = By.id("password");
    private By loginButton = By.cssSelector("[data-test='login-submit']");

    public LoginPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void enterUsername(String username) {
        driver.findElement(usernameField).sendKeys(username);
    }

    public void enterPassword(String password) {
        driver.findElement(passwordField).sendKeys(password);
    }

    public void clickLogin() {
        driver.findElement(loginButton).click();
    }

    public void loginAs(String username, String password) {
        enterUsername(username);
        enterPassword(password);
        clickLogin();

        wait.until(ExpectedConditions.urlContains("/account"));
    }

    public void loginAsAdmin(String username, String password) {
        enterUsername(username);
        enterPassword(password);
        clickLogin();

        wait.until(ExpectedConditions.urlContains("/admin"));
    }

    public boolean isLoaded() {
        WebElement element = driver.findElement(loginButton);
        return element.isDisplayed();
    }
}
