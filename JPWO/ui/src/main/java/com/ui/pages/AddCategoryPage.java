package com.ui.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class AddCategoryPage {

    private WebDriver driver;
    private WebDriverWait wait;

    private By categoryNameField = By.cssSelector("[data-test='name']");
    private By slugField = By.cssSelector("[data-test='slug']");
    private By successMessage = By.className("alert-success");

    private By submitButton = By.cssSelector("[data-test='category-submit']");

    public AddCategoryPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/admin/categories/add");
    }

    public void enterCategoryName(String name) {
        WebElement productName = wait.until(ExpectedConditions.visibilityOfElementLocated(categoryNameField));
        productName.clear();
        productName.sendKeys(name);
    }

    public void enterSlug(String slug) {
        WebElement element = driver.findElement(slugField);
        element.clear();
        element.sendKeys(slug);
    }


    public void submitForm() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(submitButton));
        button.click();

        wait.until(ExpectedConditions.visibilityOfElementLocated(successMessage));
    }

    public void fillForm(String name, String slug) {
        enterCategoryName(name);
        enterSlug(slug);
        submitForm();
    }
}
