package com.ui.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;


public class ProductAdminPage {

    private WebDriver driver;
    private WebDriverWait wait;

    private By searchField = By.cssSelector("[data-test='product-search-query']");
    private By searchButton = By.cssSelector("[data-test='product-search-submit']");


    public ProductAdminPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/admin/products");
    }

    public void search(String query) {
        WebElement searchInput = wait.until(ExpectedConditions.visibilityOfElementLocated(searchField));
        searchInput.clear();
        searchInput.sendKeys(query);
        searchInput.submit();
        driver.findElement(searchButton).click();

        wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath(String.format("//tr[td[contains(text(),'%s')]]", query))
        ));
    }

    public String getProductPrice(String productName) {
        String xpath = String.format("//tr[td[normalize-space(text())='%s']]/td[4]", productName);
        return driver.findElement(By.xpath(xpath)).getText();
    }

    public String getProductName(String productName) {
        String xpath = String.format("//tr[td[normalize-space(text())='%s']]/td[2]", productName);
        return driver.findElement(By.xpath(xpath)).getText();
    }

}
